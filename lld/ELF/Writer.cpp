//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Config.h"
#include "Filesystem.h"
#include "LinkerScript.h"
#include "MapFile.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Relocations.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/Threads.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <climits>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

namespace {
// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;

  void run();

private:
  void createSyntheticSections();
  void copyLocalSymbols();
  void addSectionSymbols();
  void addReservedSymbols();
  void forEachRelSec(std::function<void(InputSectionBase &)> Fn);
  void sortSections();
  void sortInputSections();
  void finalizeSections();
  void addPredefinedSections();
  void setReservedSymbolSections();

  std::vector<PhdrEntry *> createPhdrs();
  void removeEmptyPTLoad();
  void addPtArmExid(std::vector<PhdrEntry *> &Phdrs);
  void assignFileOffsets();
  void assignFileOffsetsBinary();
  void setPhdrs();
  void fixSectionAlignments();
  void openFile();
  void writeTrapInstr();
  void writeHeader();
  void writeSections();
  void writeSectionsBinary();
  void writeBuildId();

  std::unique_ptr<FileOutputBuffer> Buffer;

  OutputSectionFactory Factory;

  void addRelIpltSymbols();
  void addStartEndSymbols();
  void addStartStopSymbols(OutputSection *Sec);
  uint64_t getEntryAddr();
  OutputSection *findSection(StringRef Name);

  std::vector<PhdrEntry *> Phdrs;

  uint64_t FileSize;
  uint64_t SectionHeaderOff;

  bool HasGotBaseSym = false;
};
} // anonymous namespace

StringRef elf::getOutputSectionName(StringRef Name) {
  // ".zdebug_" is a prefix for ZLIB-compressed sections.
  // Because we decompressed input sections, we want to remove 'z'.
  if (Name.startswith(".zdebug_"))
    return Saver.save("." + Name.substr(2));

  if (Config->Relocatable)
    return Name;

  // This is for --emit-relocs. If .text.foo is emitted as .text, we want to
  // emit .rela.text.foo as .rel.text for consistency (this is not technically
  // required, but not doing it is odd). This code guarantees that.
  if (Name.startswith(".rel."))
    return Saver.save(".rel" + getOutputSectionName(Name.substr(4)));
  if (Name.startswith(".rela."))
    return Saver.save(".rela" + getOutputSectionName(Name.substr(5)));

  for (StringRef V :
       {".text.", ".rodata.", ".data.rel.ro.", ".data.", ".bss.rel.ro.",
        ".bss.", ".init_array.", ".fini_array.", ".ctors.", ".dtors.", ".tbss.",
        ".gcc_except_table.", ".tdata.", ".ARM.exidx.", ".ARM.extab."}) {
    StringRef Prefix = V.drop_back();
    if (Name.startswith(V) || Name == Prefix)
      return Prefix;
  }

  // CommonSection is identified as "COMMON" in linker scripts.
  // By default, it should go to .bss section.
  if (Name == "COMMON")
    return ".bss";

  return Name;
}

static bool needsInterpSection() {
  return !SharedFiles.empty() && !Config->DynamicLinker.empty() &&
         Script->needsInterpSection();
}

template <class ELFT> void elf::writeResult() { Writer<ELFT>().run(); }

template <class ELFT> void Writer<ELFT>::removeEmptyPTLoad() {
  llvm::erase_if(Phdrs, [&](const PhdrEntry *P) {
    if (P->p_type != PT_LOAD)
      return false;
    if (!P->FirstSec)
      return true;
    uint64_t Size = P->LastSec->Addr + P->LastSec->Size - P->FirstSec->Addr;
    return Size == 0;
  });
}

template <class ELFT> static void combineEhFrameSections() {
  for (InputSectionBase *&S : InputSections) {
    EhInputSection *ES = dyn_cast<EhInputSection>(S);
    if (!ES || !ES->Live)
      continue;

    InX::EhFrame->addSection<ELFT>(ES);
    S = nullptr;
  }

  std::vector<InputSectionBase *> &V = InputSections;
  V.erase(std::remove(V.begin(), V.end(), nullptr), V.end());
}

// The main function of the writer.
template <class ELFT> void Writer<ELFT>::run() {
  // Create linker-synthesized sections such as .got or .plt.
  // Such sections are of type input section.
  createSyntheticSections();

  if (!Config->Relocatable)
    combineEhFrameSections<ELFT>();

  // We need to create some reserved symbols such as _end. Create them.
  if (!Config->Relocatable)
    addReservedSymbols();

  // We want to process linker script commands. When SECTIONS command
  // is given we let it create sections.
  Script->processSectionCommands();

  // Linker scripts controls how input sections are assigned to output sections.
  // Input sections that were not handled by scripts are called "orphans", and
  // they are assigned to output sections by the default rule. Process that.
  Script->addOrphanSections(Factory);

  if (Config->Discard != DiscardPolicy::All)
    copyLocalSymbols();

  if (Config->CopyRelocs)
    addSectionSymbols();

  // Now that we have a complete set of output sections. This function
  // completes section contents. For example, we need to add strings
  // to the string table, and add entries to .got and .plt.
  // finalizeSections does that.
  finalizeSections();
  if (errorCount())
    return;

  // If -compressed-debug-sections is specified, we need to compress
  // .debug_* sections. Do it right now because it changes the size of
  // output sections.
  parallelForEach(OutputSections,
                  [](OutputSection *Sec) { Sec->maybeCompress<ELFT>(); });

  Script->assignAddresses();
  Script->allocateHeaders(Phdrs);

  // Remove empty PT_LOAD to avoid causing the dynamic linker to try to mmap a
  // 0 sized region. This has to be done late since only after assignAddresses
  // we know the size of the sections.
  removeEmptyPTLoad();

  if (!Config->OFormatBinary)
    assignFileOffsets();
  else
    assignFileOffsetsBinary();

  setPhdrs();

  if (Config->Relocatable) {
    for (OutputSection *Sec : OutputSections)
      Sec->Addr = 0;
  }

  // It does not make sense try to open the file if we have error already.
  if (errorCount())
    return;
  // Write the result down to a file.
  openFile();
  if (errorCount())
    return;

  if (!Config->OFormatBinary) {
    writeTrapInstr();
    writeHeader();
    writeSections();
  } else {
    writeSectionsBinary();
  }

  // Backfill .note.gnu.build-id section content. This is done at last
  // because the content is usually a hash value of the entire output file.
  writeBuildId();
  if (errorCount())
    return;

  // Handle -Map option.
  writeMapFile();
  if (errorCount())
    return;

  if (auto EC = Buffer->commit())
    error("failed to write to the output file: " + EC.message());
}

// Initialize Out members.
template <class ELFT> void Writer<ELFT>::createSyntheticSections() {
  // Initialize all pointers with NULL. This is needed because
  // you can call lld::elf::main more than once as a library.
  memset(&Out::First, 0, sizeof(Out));

  auto Add = [](InputSectionBase *Sec) { InputSections.push_back(Sec); };

  InX::DynStrTab = make<StringTableSection>(".dynstr", true);
  InX::Dynamic = make<DynamicSection<ELFT>>();
  if (Config->AndroidPackDynRelocs) {
    In<ELFT>::RelaDyn = make<AndroidPackedRelocationSection<ELFT>>(
        Config->IsRela ? ".rela.dyn" : ".rel.dyn");
  } else {
    In<ELFT>::RelaDyn = make<RelocationSection<ELFT>>(
        Config->IsRela ? ".rela.dyn" : ".rel.dyn", Config->ZCombreloc);
  }
  InX::ShStrTab = make<StringTableSection>(".shstrtab", false);

  Out::ElfHeader = make<OutputSection>("", 0, SHF_ALLOC);
  Out::ElfHeader->Size = sizeof(Elf_Ehdr);
  Out::ProgramHeaders = make<OutputSection>("", 0, SHF_ALLOC);
  Out::ProgramHeaders->Alignment = Config->Wordsize;

  if (needsInterpSection()) {
    InX::Interp = createInterpSection();
    Add(InX::Interp);
  } else {
    InX::Interp = nullptr;
  }

  if (Config->Strip != StripPolicy::All) {
    InX::StrTab = make<StringTableSection>(".strtab", false);
    InX::SymTab = make<SymbolTableSection<ELFT>>(*InX::StrTab);
  }

  if (Config->BuildId != BuildIdKind::None) {
    InX::BuildId = make<BuildIdSection>();
    Add(InX::BuildId);
  }

  InX::Bss = make<BssSection>(".bss", 0, 1);
  Add(InX::Bss);
  InX::BssRelRo = make<BssSection>(".bss.rel.ro", 0, 1);
  Add(InX::BssRelRo);

  // Add MIPS-specific sections.
  if (Config->EMachine == EM_MIPS) {
    if (!Config->Shared && Config->HasDynSymTab) {
      InX::MipsRldMap = make<MipsRldMapSection>();
      Add(InX::MipsRldMap);
    }
    if (auto *Sec = MipsAbiFlagsSection<ELFT>::create())
      Add(Sec);
    if (auto *Sec = MipsOptionsSection<ELFT>::create())
      Add(Sec);
    if (auto *Sec = MipsReginfoSection<ELFT>::create())
      Add(Sec);
  }

  if (Config->HasDynSymTab) {
    InX::DynSymTab = make<SymbolTableSection<ELFT>>(*InX::DynStrTab);
    Add(InX::DynSymTab);

    In<ELFT>::VerSym = make<VersionTableSection<ELFT>>();
    Add(In<ELFT>::VerSym);

    if (!Config->VersionDefinitions.empty()) {
      In<ELFT>::VerDef = make<VersionDefinitionSection<ELFT>>();
      Add(In<ELFT>::VerDef);
    }

    In<ELFT>::VerNeed = make<VersionNeedSection<ELFT>>();
    Add(In<ELFT>::VerNeed);

    if (Config->GnuHash) {
      InX::GnuHashTab = make<GnuHashTableSection>();
      Add(InX::GnuHashTab);
    }

    if (Config->SysvHash) {
      InX::HashTab = make<HashTableSection>();
      Add(InX::HashTab);
    }

    Add(InX::Dynamic);
    Add(InX::DynStrTab);
    Add(In<ELFT>::RelaDyn);
  }

  // Add .got. MIPS' .got is so different from the other archs,
  // it has its own class.
  if (Config->EMachine == EM_MIPS) {
    InX::MipsGot = make<MipsGotSection>();
    Add(InX::MipsGot);
  } else {
    InX::Got = make<GotSection>();
    Add(InX::Got);
  }

  InX::GotPlt = make<GotPltSection>();
  Add(InX::GotPlt);
  InX::IgotPlt = make<IgotPltSection>();
  Add(InX::IgotPlt);

  if (Config->GdbIndex) {
    InX::GdbIndex = createGdbIndex<ELFT>();
    Add(InX::GdbIndex);
  }

  // We always need to add rel[a].plt to output if it has entries.
  // Even for static linking it can contain R_[*]_IRELATIVE relocations.
  In<ELFT>::RelaPlt = make<RelocationSection<ELFT>>(
      Config->IsRela ? ".rela.plt" : ".rel.plt", false /*Sort*/);
  Add(In<ELFT>::RelaPlt);

  // The RelaIplt immediately follows .rel.plt (.rel.dyn for ARM) to ensure
  // that the IRelative relocations are processed last by the dynamic loader.
  // We cannot place the iplt section in .rel.dyn when Android relocation
  // packing is enabled because that would cause a section type mismatch.
  // However, because the Android dynamic loader reads .rel.plt after .rel.dyn,
  // we can get the desired behaviour by placing the iplt section in .rel.plt.
  In<ELFT>::RelaIplt = make<RelocationSection<ELFT>>(
      (Config->EMachine == EM_ARM && !Config->AndroidPackDynRelocs)
          ? ".rel.dyn"
          : In<ELFT>::RelaPlt->Name,
      false /*Sort*/);
  Add(In<ELFT>::RelaIplt);

  InX::Plt = make<PltSection>(Target->PltHeaderSize);
  Add(InX::Plt);
  InX::Iplt = make<PltSection>(0);
  Add(InX::Iplt);

  if (!Config->Relocatable) {
    if (Config->EhFrameHdr) {
      InX::EhFrameHdr = make<EhFrameHeader>();
      Add(InX::EhFrameHdr);
    }
    InX::EhFrame = make<EhFrameSection>();
    Add(InX::EhFrame);
  }

  if (InX::SymTab)
    Add(InX::SymTab);
  Add(InX::ShStrTab);
  if (InX::StrTab)
    Add(InX::StrTab);
}

static bool shouldKeepInSymtab(SectionBase *Sec, StringRef SymName,
                               const SymbolBody &B) {
  if (B.isFile() || B.isSection())
    return false;

  // If sym references a section in a discarded group, don't keep it.
  if (Sec == &InputSection::Discarded)
    return false;

  if (Config->Discard == DiscardPolicy::None)
    return true;

  // In ELF assembly .L symbols are normally discarded by the assembler.
  // If the assembler fails to do so, the linker discards them if
  // * --discard-locals is used.
  // * The symbol is in a SHF_MERGE section, which is normally the reason for
  //   the assembler keeping the .L symbol.
  if (!SymName.startswith(".L") && !SymName.empty())
    return true;

  if (Config->Discard == DiscardPolicy::Locals)
    return false;

  return !Sec || !(Sec->Flags & SHF_MERGE);
}

static bool includeInSymtab(const SymbolBody &B) {
  if (!B.isLocal() && !B.symbol()->IsUsedInRegularObj)
    return false;

  if (auto *D = dyn_cast<DefinedRegular>(&B)) {
    // Always include absolute symbols.
    SectionBase *Sec = D->Section;
    if (!Sec)
      return true;
    if (auto *IS = dyn_cast<InputSectionBase>(Sec)) {
      Sec = IS->Repl;
      IS = cast<InputSectionBase>(Sec);
      // Exclude symbols pointing to garbage-collected sections.
      if (!IS->Live)
        return false;
    }
    if (auto *S = dyn_cast<MergeInputSection>(Sec))
      if (!S->getSectionPiece(D->Value)->Live)
        return false;
  }
  return true;
}

// Local symbols are not in the linker's symbol table. This function scans
// each object file's symbol table to copy local symbols to the output.
template <class ELFT> void Writer<ELFT>::copyLocalSymbols() {
  if (!InX::SymTab)
    return;
  for (InputFile *File : ObjectFiles) {
    ObjFile<ELFT> *F = cast<ObjFile<ELFT>>(File);
    for (SymbolBody *B : F->getLocalSymbols()) {
      if (!B->isLocal())
        fatal(toString(F) +
              ": broken object: getLocalSymbols returns a non-local symbol");
      auto *DR = dyn_cast<DefinedRegular>(B);

      // No reason to keep local undefined symbol in symtab.
      if (!DR)
        continue;
      if (!includeInSymtab(*B))
        continue;

      SectionBase *Sec = DR->Section;
      if (!shouldKeepInSymtab(Sec, B->getName(), *B))
        continue;
      InX::SymTab->addSymbol(B);
    }
  }
}

template <class ELFT> void Writer<ELFT>::addSectionSymbols() {
  // Create one STT_SECTION symbol for each output section we might
  // have a relocation with.
  for (BaseCommand *Base : Script->SectionCommands) {
    auto *Sec = dyn_cast<OutputSection>(Base);
    if (!Sec)
      continue;
    auto I = llvm::find_if(Sec->SectionCommands, [](BaseCommand *Base) {
      if (auto *ISD = dyn_cast<InputSectionDescription>(Base))
        return !ISD->Sections.empty();
      return false;
    });
    if (I == Sec->SectionCommands.end())
      continue;
    InputSection *IS = cast<InputSectionDescription>(*I)->Sections[0];
    if (isa<SyntheticSection>(IS) || IS->Type == SHT_REL ||
        IS->Type == SHT_RELA)
      continue;

    auto *Sym =
        make<DefinedRegular>("", /*IsLocal=*/true, /*StOther=*/0, STT_SECTION,
                             /*Value=*/0, /*Size=*/0, IS);
    InX::SymTab->addSymbol(Sym);
  }
}

// Today's loaders have a feature to make segments read-only after
// processing dynamic relocations to enhance security. PT_GNU_RELRO
// is defined for that.
//
// This function returns true if a section needs to be put into a
// PT_GNU_RELRO segment.
static bool isRelroSection(const OutputSection *Sec) {
  if (!Config->ZRelro)
    return false;

  uint64_t Flags = Sec->Flags;

  // Non-allocatable or non-writable sections don't need RELRO because
  // they are not writable or not even mapped to memory in the first place.
  // RELRO is for sections that are essentially read-only but need to
  // be writable only at process startup to allow dynamic linker to
  // apply relocations.
  if (!(Flags & SHF_ALLOC) || !(Flags & SHF_WRITE))
    return false;

  // Once initialized, TLS data segments are used as data templates
  // for a thread-local storage. For each new thread, runtime
  // allocates memory for a TLS and copy templates there. No thread
  // are supposed to use templates directly. Thus, it can be in RELRO.
  if (Flags & SHF_TLS)
    return true;

  // .init_array, .preinit_array and .fini_array contain pointers to
  // functions that are executed on process startup or exit. These
  // pointers are set by the static linker, and they are not expected
  // to change at runtime. But if you are an attacker, you could do
  // interesting things by manipulating pointers in .fini_array, for
  // example. So they are put into RELRO.
  uint32_t Type = Sec->Type;
  if (Type == SHT_INIT_ARRAY || Type == SHT_FINI_ARRAY ||
      Type == SHT_PREINIT_ARRAY)
    return true;

  // .got contains pointers to external symbols. They are resolved by
  // the dynamic linker when a module is loaded into memory, and after
  // that they are not expected to change. So, it can be in RELRO.
  if (InX::Got && Sec == InX::Got->getParent())
    return true;

  // .got.plt contains pointers to external function symbols. They are
  // by default resolved lazily, so we usually cannot put it into RELRO.
  // However, if "-z now" is given, the lazy symbol resolution is
  // disabled, which enables us to put it into RELRO.
  if (Sec == InX::GotPlt->getParent())
    return Config->ZNow;

  // .dynamic section contains data for the dynamic linker, and
  // there's no need to write to it at runtime, so it's better to put
  // it into RELRO.
  if (Sec == InX::Dynamic->getParent())
    return true;

  // .bss.rel.ro is used for copy relocations for read-only symbols.
  // Since the dynamic linker needs to process copy relocations, the
  // section cannot be read-only, but once initialized, they shouldn't
  // change.
  if (Sec == InX::BssRelRo->getParent())
    return true;

  // Sections with some special names are put into RELRO. This is a
  // bit unfortunate because section names shouldn't be significant in
  // ELF in spirit. But in reality many linker features depend on
  // magic section names.
  StringRef S = Sec->Name;
  return S == ".data.rel.ro" || S == ".ctors" || S == ".dtors" || S == ".jcr" ||
         S == ".eh_frame" || S == ".openbsd.randomdata";
}

// We compute a rank for each section. The rank indicates where the
// section should be placed in the file.  Instead of using simple
// numbers (0,1,2...), we use a series of flags. One for each decision
// point when placing the section.
// Using flags has two key properties:
// * It is easy to check if a give branch was taken.
// * It is easy two see how similar two ranks are (see getRankProximity).
enum RankFlags {
  RF_NOT_ADDR_SET = 1 << 16,
  RF_NOT_INTERP = 1 << 15,
  RF_NOT_ALLOC = 1 << 14,
  RF_WRITE = 1 << 13,
  RF_EXEC_WRITE = 1 << 12,
  RF_EXEC = 1 << 11,
  RF_NON_TLS_BSS = 1 << 10,
  RF_NON_TLS_BSS_RO = 1 << 9,
  RF_NOT_TLS = 1 << 8,
  RF_BSS = 1 << 7,
  RF_PPC_NOT_TOCBSS = 1 << 6,
  RF_PPC_OPD = 1 << 5,
  RF_PPC_TOCL = 1 << 4,
  RF_PPC_TOC = 1 << 3,
  RF_PPC_BRANCH_LT = 1 << 2,
  RF_MIPS_GPREL = 1 << 1,
  RF_MIPS_NOT_GOT = 1 << 0
};

static unsigned getSectionRank(const OutputSection *Sec) {
  unsigned Rank = 0;

  // We want to put section specified by -T option first, so we
  // can start assigning VA starting from them later.
  if (Config->SectionStartMap.count(Sec->Name))
    return Rank;
  Rank |= RF_NOT_ADDR_SET;

  // Put .interp first because some loaders want to see that section
  // on the first page of the executable file when loaded into memory.
  if (Sec->Name == ".interp")
    return Rank;
  Rank |= RF_NOT_INTERP;

  // Allocatable sections go first to reduce the total PT_LOAD size and
  // so debug info doesn't change addresses in actual code.
  if (!(Sec->Flags & SHF_ALLOC))
    return Rank | RF_NOT_ALLOC;

  // Sort sections based on their access permission in the following
  // order: R, RX, RWX, RW.  This order is based on the following
  // considerations:
  // * Read-only sections come first such that they go in the
  //   PT_LOAD covering the program headers at the start of the file.
  // * Read-only, executable sections come next, unless the
  //   -no-rosegment option is used.
  // * Writable, executable sections follow such that .plt on
  //   architectures where it needs to be writable will be placed
  //   between .text and .data.
  // * Writable sections come last, such that .bss lands at the very
  //   end of the last PT_LOAD.
  bool IsExec = Sec->Flags & SHF_EXECINSTR;
  bool IsWrite = Sec->Flags & SHF_WRITE;

  if (IsExec) {
    if (IsWrite)
      Rank |= RF_EXEC_WRITE;
    else if (!Config->SingleRoRx)
      Rank |= RF_EXEC;
  } else {
    if (IsWrite)
      Rank |= RF_WRITE;
  }

  // If we got here we know that both A and B are in the same PT_LOAD.

  bool IsTls = Sec->Flags & SHF_TLS;
  bool IsNoBits = Sec->Type == SHT_NOBITS;

  // The first requirement we have is to put (non-TLS) nobits sections last. The
  // reason is that the only thing the dynamic linker will see about them is a
  // p_memsz that is larger than p_filesz. Seeing that it zeros the end of the
  // PT_LOAD, so that has to correspond to the nobits sections.
  bool IsNonTlsNoBits = IsNoBits && !IsTls;
  if (IsNonTlsNoBits)
    Rank |= RF_NON_TLS_BSS;

  // We place nobits RelRo sections before plain r/w ones, and non-nobits RelRo
  // sections after r/w ones, so that the RelRo sections are contiguous.
  bool IsRelRo = isRelroSection(Sec);
  if (IsNonTlsNoBits && !IsRelRo)
    Rank |= RF_NON_TLS_BSS_RO;
  if (!IsNonTlsNoBits && IsRelRo)
    Rank |= RF_NON_TLS_BSS_RO;

  // The TLS initialization block needs to be a single contiguous block in a R/W
  // PT_LOAD, so stick TLS sections directly before the other RelRo R/W
  // sections. The TLS NOBITS sections are placed here as they don't take up
  // virtual address space in the PT_LOAD.
  if (!IsTls)
    Rank |= RF_NOT_TLS;

  // Within the TLS initialization block, the non-nobits sections need to appear
  // first.
  if (IsNoBits)
    Rank |= RF_BSS;

  // Some architectures have additional ordering restrictions for sections
  // within the same PT_LOAD.
  if (Config->EMachine == EM_PPC64) {
    // PPC64 has a number of special SHT_PROGBITS+SHF_ALLOC+SHF_WRITE sections
    // that we would like to make sure appear is a specific order to maximize
    // their coverage by a single signed 16-bit offset from the TOC base
    // pointer. Conversely, the special .tocbss section should be first among
    // all SHT_NOBITS sections. This will put it next to the loaded special
    // PPC64 sections (and, thus, within reach of the TOC base pointer).
    StringRef Name = Sec->Name;
    if (Name != ".tocbss")
      Rank |= RF_PPC_NOT_TOCBSS;

    if (Name == ".opd")
      Rank |= RF_PPC_OPD;

    if (Name == ".toc1")
      Rank |= RF_PPC_TOCL;

    if (Name == ".toc")
      Rank |= RF_PPC_TOC;

    if (Name == ".branch_lt")
      Rank |= RF_PPC_BRANCH_LT;
  }
  if (Config->EMachine == EM_MIPS) {
    // All sections with SHF_MIPS_GPREL flag should be grouped together
    // because data in these sections is addressable with a gp relative address.
    if (Sec->Flags & SHF_MIPS_GPREL)
      Rank |= RF_MIPS_GPREL;

    if (Sec->Name != ".got")
      Rank |= RF_MIPS_NOT_GOT;
  }

  return Rank;
}

static bool compareSections(const BaseCommand *ACmd, const BaseCommand *BCmd) {
  const OutputSection *A = cast<OutputSection>(ACmd);
  const OutputSection *B = cast<OutputSection>(BCmd);
  if (A->SortRank != B->SortRank)
    return A->SortRank < B->SortRank;
  if (!(A->SortRank & RF_NOT_ADDR_SET))
    return Config->SectionStartMap.lookup(A->Name) <
           Config->SectionStartMap.lookup(B->Name);
  return false;
}

void PhdrEntry::add(OutputSection *Sec) {
  LastSec = Sec;
  if (!FirstSec)
    FirstSec = Sec;
  p_align = std::max(p_align, Sec->Alignment);
  if (p_type == PT_LOAD)
    Sec->PtLoad = this;
}

template <class ELFT>
static DefinedRegular *
addOptionalRegular(StringRef Name, SectionBase *Sec, uint64_t Val,
                   uint8_t StOther = STV_HIDDEN, uint8_t Binding = STB_GLOBAL) {
  SymbolBody *S = Symtab->find(Name);
  if (!S || S->isInCurrentOutput())
    return nullptr;
  Symbol *Sym = Symtab->addRegular<ELFT>(Name, StOther, STT_NOTYPE, Val,
                                         /*Size=*/0, Binding, Sec,
                                         /*File=*/nullptr);
  return cast<DefinedRegular>(Sym->body());
}

// The beginning and the ending of .rel[a].plt section are marked
// with __rel[a]_iplt_{start,end} symbols if it is a statically linked
// executable. The runtime needs these symbols in order to resolve
// all IRELATIVE relocs on startup. For dynamic executables, we don't
// need these symbols, since IRELATIVE relocs are resolved through GOT
// and PLT. For details, see http://www.airs.com/blog/archives/403.
template <class ELFT> void Writer<ELFT>::addRelIpltSymbols() {
  if (!Config->Static)
    return;
  StringRef S = Config->IsRela ? "__rela_iplt_start" : "__rel_iplt_start";
  addOptionalRegular<ELFT>(S, In<ELFT>::RelaIplt, 0, STV_HIDDEN, STB_WEAK);

  S = Config->IsRela ? "__rela_iplt_end" : "__rel_iplt_end";
  addOptionalRegular<ELFT>(S, In<ELFT>::RelaIplt, -1, STV_HIDDEN, STB_WEAK);
}

// The linker is expected to define some symbols depending on
// the linking result. This function defines such symbols.
template <class ELFT> void Writer<ELFT>::addReservedSymbols() {
  if (Config->EMachine == EM_MIPS) {
    // Define _gp for MIPS. st_value of _gp symbol will be updated by Writer
    // so that it points to an absolute address which by default is relative
    // to GOT. Default offset is 0x7ff0.
    // See "Global Data Symbols" in Chapter 6 in the following document:
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    ElfSym::MipsGp = Symtab->addAbsolute<ELFT>("_gp", STV_HIDDEN, STB_LOCAL);

    // On MIPS O32 ABI, _gp_disp is a magic symbol designates offset between
    // start of function and 'gp' pointer into GOT.
    if (Symtab->find("_gp_disp"))
      ElfSym::MipsGpDisp =
          Symtab->addAbsolute<ELFT>("_gp_disp", STV_HIDDEN, STB_LOCAL);

    // The __gnu_local_gp is a magic symbol equal to the current value of 'gp'
    // pointer. This symbol is used in the code generated by .cpload pseudo-op
    // in case of using -mno-shared option.
    // https://sourceware.org/ml/binutils/2004-12/msg00094.html
    if (Symtab->find("__gnu_local_gp"))
      ElfSym::MipsLocalGp =
          Symtab->addAbsolute<ELFT>("__gnu_local_gp", STV_HIDDEN, STB_LOCAL);
  }

  // The _GLOBAL_OFFSET_TABLE_ symbol is defined by target convention to
  // be at some offset from the base of the .got section, usually 0 or the end
  // of the .got
  InputSection *GotSection = InX::MipsGot ? cast<InputSection>(InX::MipsGot)
                                          : cast<InputSection>(InX::Got);
  ElfSym::GlobalOffsetTable = addOptionalRegular<ELFT>(
      "_GLOBAL_OFFSET_TABLE_", GotSection, Target->GotBaseSymOff);

  // __ehdr_start is the location of ELF file headers. Note that we define
  // this symbol unconditionally even when using a linker script, which
  // differs from the behavior implemented by GNU linker which only define
  // this symbol if ELF headers are in the memory mapped segment.
  // __executable_start is not documented, but the expectation of at
  // least the android libc is that it points to the elf header too.
  // __dso_handle symbol is passed to cxa_finalize as a marker to identify
  // each DSO. The address of the symbol doesn't matter as long as they are
  // different in different DSOs, so we chose the start address of the DSO.
  for (const char *Name :
       {"__ehdr_start", "__executable_start", "__dso_handle"})
    addOptionalRegular<ELFT>(Name, Out::ElfHeader, 0, STV_HIDDEN);

  // If linker script do layout we do not need to create any standart symbols.
  if (Script->HasSectionsCommand)
    return;

  auto Add = [](StringRef S, int64_t Pos) {
    return addOptionalRegular<ELFT>(S, Out::ElfHeader, Pos, STV_DEFAULT);
  };

  ElfSym::Bss = Add("__bss_start", 0);
  ElfSym::End1 = Add("end", -1);
  ElfSym::End2 = Add("_end", -1);
  ElfSym::Etext1 = Add("etext", -1);
  ElfSym::Etext2 = Add("_etext", -1);
  ElfSym::Edata1 = Add("edata", -1);
  ElfSym::Edata2 = Add("_edata", -1);
}

template <class ELFT>
void Writer<ELFT>::forEachRelSec(std::function<void(InputSectionBase &)> Fn) {
  // Scan all relocations. Each relocation goes through a series
  // of tests to determine if it needs special treatment, such as
  // creating GOT, PLT, copy relocations, etc.
  // Note that relocations for non-alloc sections are directly
  // processed by InputSection::relocateNonAlloc.
  for (InputSectionBase *IS : InputSections)
    if (IS->Live && isa<InputSection>(IS) && (IS->Flags & SHF_ALLOC))
      Fn(*IS);
  for (EhInputSection *ES : InX::EhFrame->Sections)
    Fn(*ES);
}

// This function generates assignments for predefined symbols (e.g. _end or
// _etext) and inserts them into the commands sequence to be processed at the
// appropriate time. This ensures that the value is going to be correct by the
// time any references to these symbols are processed and is equivalent to
// defining these symbols explicitly in the linker script.
template <class ELFT> void Writer<ELFT>::setReservedSymbolSections() {
  PhdrEntry *Last = nullptr;
  PhdrEntry *LastRO = nullptr;
  PhdrEntry *LastRW = nullptr;

  for (PhdrEntry *P : Phdrs) {
    if (P->p_type != PT_LOAD)
      continue;
    Last = P;
    if (P->p_flags & PF_W)
      LastRW = P;
    else
      LastRO = P;
  }

  if (LastRO) {
    // _etext is the first location after the last read-only loadable segment.
    if (ElfSym::Etext1)
      ElfSym::Etext1->Section = LastRO->LastSec;
    if (ElfSym::Etext2)
      ElfSym::Etext2->Section = LastRO->LastSec;
  }

  if (Last) {
    // _edata points to the end of the last mapped initialized section.
    OutputSection *Edata = nullptr;
    for (OutputSection *OS : OutputSections) {
      if (OS->Type != SHT_NOBITS)
        Edata = OS;
      if (OS == Last->LastSec)
        break;
    }

    if (ElfSym::Edata1)
      ElfSym::Edata1->Section = Edata;
    if (ElfSym::Edata2)
      ElfSym::Edata2->Section = Edata;

    // _end is the first location after the uninitialized data region.
    if (ElfSym::End1)
      ElfSym::End1->Section = Last->LastSec;
    if (ElfSym::End2)
      ElfSym::End2->Section = Last->LastSec;
  }

  if (ElfSym::Bss)
    ElfSym::Bss->Section = findSection(".bss");

  // Setup MIPS _gp_disp/__gnu_local_gp symbols which should
  // be equal to the _gp symbol's value.
  if (ElfSym::MipsGp) {
    // Find GP-relative section with the lowest address
    // and use this address to calculate default _gp value.
    for (OutputSection *OS : OutputSections) {
      if (OS->Flags & SHF_MIPS_GPREL) {
        ElfSym::MipsGp->Section = OS;
        ElfSym::MipsGp->Value = 0x7ff0;
        break;
      }
    }
  }
}

// We want to find how similar two ranks are.
// The more branches in getSectionRank that match, the more similar they are.
// Since each branch corresponds to a bit flag, we can just use
// countLeadingZeros.
static int getRankProximityAux(OutputSection *A, OutputSection *B) {
  return countLeadingZeros(A->SortRank ^ B->SortRank);
}

static int getRankProximity(OutputSection *A, BaseCommand *B) {
  if (auto *Sec = dyn_cast<OutputSection>(B))
    if (Sec->Live)
      return getRankProximityAux(A, Sec);
  return -1;
}

// When placing orphan sections, we want to place them after symbol assignments
// so that an orphan after
//   begin_foo = .;
//   foo : { *(foo) }
//   end_foo = .;
// doesn't break the intended meaning of the begin/end symbols.
// We don't want to go over sections since findOrphanPos is the
// one in charge of deciding the order of the sections.
// We don't want to go over changes to '.', since doing so in
//  rx_sec : { *(rx_sec) }
//  . = ALIGN(0x1000);
//  /* The RW PT_LOAD starts here*/
//  rw_sec : { *(rw_sec) }
// would mean that the RW PT_LOAD would become unaligned.
static bool shouldSkip(BaseCommand *Cmd) {
  if (isa<OutputSection>(Cmd))
    return false;
  if (auto *Assign = dyn_cast<SymbolAssignment>(Cmd))
    return Assign->Name != ".";
  return true;
}

// We want to place orphan sections so that they share as much
// characteristics with their neighbors as possible. For example, if
// both are rw, or both are tls.
template <typename ELFT>
static std::vector<BaseCommand *>::iterator
findOrphanPos(std::vector<BaseCommand *>::iterator B,
              std::vector<BaseCommand *>::iterator E) {
  OutputSection *Sec = cast<OutputSection>(*E);

  // Find the first element that has as close a rank as possible.
  auto I = std::max_element(B, E, [=](BaseCommand *A, BaseCommand *B) {
    return getRankProximity(Sec, A) < getRankProximity(Sec, B);
  });
  if (I == E)
    return E;

  // Consider all existing sections with the same proximity.
  int Proximity = getRankProximity(Sec, *I);
  for (; I != E; ++I) {
    auto *CurSec = dyn_cast<OutputSection>(*I);
    if (!CurSec || !CurSec->Live)
      continue;
    if (getRankProximity(Sec, CurSec) != Proximity ||
        Sec->SortRank < CurSec->SortRank)
      break;
  }

  auto IsLiveSection = [](BaseCommand *Cmd) {
    auto *OS = dyn_cast<OutputSection>(Cmd);
    return OS && OS->Live;
  };

  auto J = std::find_if(llvm::make_reverse_iterator(I),
                        llvm::make_reverse_iterator(B), IsLiveSection);
  I = J.base();

  // As a special case, if the orphan section is the last section, put
  // it at the very end, past any other commands.
  // This matches bfd's behavior and is convenient when the linker script fully
  // specifies the start of the file, but doesn't care about the end (the non
  // alloc sections for example).
  auto NextSec = std::find_if(I, E, IsLiveSection);
  if (NextSec == E)
    return E;

  while (I != E && shouldSkip(*I))
    ++I;
  return I;
}

// If no layout was provided by linker script, we want to apply default
// sorting for special input sections and handle --symbol-ordering-file.
template <class ELFT> void Writer<ELFT>::sortInputSections() {
  assert(!Script->HasSectionsCommand);

  // Sort input sections by priority using the list provided
  // by --symbol-ordering-file.
  DenseMap<SectionBase *, int> Order = buildSectionOrder();
  if (!Order.empty())
    for (BaseCommand *Base : Script->SectionCommands)
      if (auto *Sec = dyn_cast<OutputSection>(Base))
        if (Sec->Live)
          Sec->sort([&](InputSectionBase *S) { return Order.lookup(S); });

  // Sort input sections by section name suffixes for
  // __attribute__((init_priority(N))).
  if (OutputSection *Sec = findSection(".init_array"))
    Sec->sortInitFini();
  if (OutputSection *Sec = findSection(".fini_array"))
    Sec->sortInitFini();

  // Sort input sections by the special rule for .ctors and .dtors.
  if (OutputSection *Sec = findSection(".ctors"))
    Sec->sortCtorsDtors();
  if (OutputSection *Sec = findSection(".dtors"))
    Sec->sortCtorsDtors();
}

template <class ELFT> void Writer<ELFT>::sortSections() {
  Script->adjustSectionsBeforeSorting();

  // Don't sort if using -r. It is not necessary and we want to preserve the
  // relative order for SHF_LINK_ORDER sections.
  if (Config->Relocatable)
    return;

  for (BaseCommand *Base : Script->SectionCommands)
    if (auto *Sec = dyn_cast<OutputSection>(Base))
      Sec->SortRank = getSectionRank(Sec);

  if (!Script->HasSectionsCommand) {
    sortInputSections();

    // We know that all the OutputSections are contiguous in this case.
    auto E = Script->SectionCommands.end();
    auto I = Script->SectionCommands.begin();
    auto IsSection = [](BaseCommand *Base) { return isa<OutputSection>(Base); };
    I = std::find_if(I, E, IsSection);
    E = std::find_if(llvm::make_reverse_iterator(E),
                     llvm::make_reverse_iterator(I), IsSection)
            .base();
    std::stable_sort(I, E, compareSections);
    return;
  }

  // Orphan sections are sections present in the input files which are
  // not explicitly placed into the output file by the linker script.
  //
  // The sections in the linker script are already in the correct
  // order. We have to figuere out where to insert the orphan
  // sections.
  //
  // The order of the sections in the script is arbitrary and may not agree with
  // compareSections. This means that we cannot easily define a strict weak
  // ordering. To see why, consider a comparison of a section in the script and
  // one not in the script. We have a two simple options:
  // * Make them equivalent (a is not less than b, and b is not less than a).
  //   The problem is then that equivalence has to be transitive and we can
  //   have sections a, b and c with only b in a script and a less than c
  //   which breaks this property.
  // * Use compareSectionsNonScript. Given that the script order doesn't have
  //   to match, we can end up with sections a, b, c, d where b and c are in the
  //   script and c is compareSectionsNonScript less than b. In which case d
  //   can be equivalent to c, a to b and d < a. As a concrete example:
  //   .a (rx) # not in script
  //   .b (rx) # in script
  //   .c (ro) # in script
  //   .d (ro) # not in script
  //
  // The way we define an order then is:
  // *  Sort only the orphan sections. They are in the end right now.
  // *  Move each orphan section to its preferred position. We try
  //    to put each section in the last position where it it can share
  //    a PT_LOAD.
  //
  // There is some ambiguity as to where exactly a new entry should be
  // inserted, because Commands contains not only output section
  // commands but also other types of commands such as symbol assignment
  // expressions. There's no correct answer here due to the lack of the
  // formal specification of the linker script. We use heuristics to
  // determine whether a new output command should be added before or
  // after another commands. For the details, look at shouldSkip
  // function.

  auto I = Script->SectionCommands.begin();
  auto E = Script->SectionCommands.end();
  auto NonScriptI = std::find_if(I, E, [](BaseCommand *Base) {
    if (auto *Sec = dyn_cast<OutputSection>(Base))
      return Sec->Live && Sec->SectionIndex == INT_MAX;
    return false;
  });

  // Sort the orphan sections.
  std::stable_sort(NonScriptI, E, compareSections);

  // As a horrible special case, skip the first . assignment if it is before any
  // section. We do this because it is common to set a load address by starting
  // the script with ". = 0xabcd" and the expectation is that every section is
  // after that.
  auto FirstSectionOrDotAssignment =
      std::find_if(I, E, [](BaseCommand *Cmd) { return !shouldSkip(Cmd); });
  if (FirstSectionOrDotAssignment != E &&
      isa<SymbolAssignment>(**FirstSectionOrDotAssignment))
    ++FirstSectionOrDotAssignment;
  I = FirstSectionOrDotAssignment;

  while (NonScriptI != E) {
    auto Pos = findOrphanPos<ELFT>(I, NonScriptI);
    OutputSection *Orphan = cast<OutputSection>(*NonScriptI);

    // As an optimization, find all sections with the same sort rank
    // and insert them with one rotate.
    unsigned Rank = Orphan->SortRank;
    auto End = std::find_if(NonScriptI + 1, E, [=](BaseCommand *Cmd) {
      return cast<OutputSection>(Cmd)->SortRank != Rank;
    });
    std::rotate(Pos, NonScriptI, End);
    NonScriptI = End;
  }

  Script->adjustSectionsAfterSorting();
}

static void applySynthetic(const std::vector<SyntheticSection *> &Sections,
                           std::function<void(SyntheticSection *)> Fn) {
  for (SyntheticSection *SS : Sections)
    if (SS && SS->getParent() && !SS->empty())
      Fn(SS);
}

// In order to allow users to manipulate linker-synthesized sections,
// we had to add synthetic sections to the input section list early,
// even before we make decisions whether they are needed. This allows
// users to write scripts like this: ".mygot : { .got }".
//
// Doing it has an unintended side effects. If it turns out that we
// don't need a .got (for example) at all because there's no
// relocation that needs a .got, we don't want to emit .got.
//
// To deal with the above problem, this function is called after
// scanRelocations is called to remove synthetic sections that turn
// out to be empty.
static void removeUnusedSyntheticSections() {
  // All input synthetic sections that can be empty are placed after
  // all regular ones. We iterate over them all and exit at first
  // non-synthetic.
  for (InputSectionBase *S : llvm::reverse(InputSections)) {
    SyntheticSection *SS = dyn_cast<SyntheticSection>(S);
    if (!SS)
      return;
    OutputSection *OS = SS->getParent();
    if (!SS->empty() || !OS)
      continue;

    std::vector<BaseCommand *>::iterator Empty = OS->SectionCommands.end();
    for (auto I = OS->SectionCommands.begin(), E = OS->SectionCommands.end();
         I != E; ++I) {
      BaseCommand *B = *I;
      if (auto *ISD = dyn_cast<InputSectionDescription>(B)) {
        llvm::erase_if(ISD->Sections,
                       [=](InputSection *IS) { return IS == SS; });
        if (ISD->Sections.empty())
          Empty = I;
      }
    }
    if (Empty != OS->SectionCommands.end())
      OS->SectionCommands.erase(Empty);

    // If there are no other sections in the output section, remove it from the
    // output.
    if (OS->SectionCommands.empty())
      OS->Live = false;
  }
}

// Returns true if a symbol can be replaced at load-time by a symbol
// with the same name defined in other ELF executable or DSO.
static bool computeIsPreemptible(const SymbolBody &B) {
  assert(!B.isLocal());
  // Only symbols that appear in dynsym can be preempted.
  if (!B.symbol()->includeInDynsym())
    return false;

  // Only default visibility symbols can be preempted.
  if (B.symbol()->Visibility != STV_DEFAULT)
    return false;

  // At this point copy relocations have not been created yet, so any
  // symbol that is not defined locally is preemptible.
  if (!B.isInCurrentOutput())
    return true;

  // If we have a dynamic list it specifies which local symbols are preemptible.
  if (Config->HasDynamicList)
    return false;

  if (!Config->Shared)
    return false;

  // -Bsymbolic means that definitions are not preempted.
  if (Config->Bsymbolic || (Config->BsymbolicFunctions && B.isFunc()))
    return false;
  return true;
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::finalizeSections() {
  Out::DebugInfo = findSection(".debug_info");
  Out::PreinitArray = findSection(".preinit_array");
  Out::InitArray = findSection(".init_array");
  Out::FiniArray = findSection(".fini_array");

  // The linker needs to define SECNAME_start, SECNAME_end and SECNAME_stop
  // symbols for sections, so that the runtime can get the start and end
  // addresses of each section by section name. Add such symbols.
  if (!Config->Relocatable) {
    addStartEndSymbols();
    for (BaseCommand *Base : Script->SectionCommands)
      if (auto *Sec = dyn_cast<OutputSection>(Base))
        addStartStopSymbols(Sec);
  }

  // Add _DYNAMIC symbol. Unlike GNU gold, our _DYNAMIC symbol has no type.
  // It should be okay as no one seems to care about the type.
  // Even the author of gold doesn't remember why gold behaves that way.
  // https://sourceware.org/ml/binutils/2002-03/msg00360.html
  if (InX::DynSymTab)
    Symtab->addRegular<ELFT>("_DYNAMIC", STV_HIDDEN, STT_NOTYPE, 0 /*Value*/,
                             /*Size=*/0, STB_WEAK, InX::Dynamic,
                             /*File=*/nullptr);

  // Define __rel[a]_iplt_{start,end} symbols if needed.
  addRelIpltSymbols();

  // This responsible for splitting up .eh_frame section into
  // pieces. The relocation scan uses those pieces, so this has to be
  // earlier.
  applySynthetic({InX::EhFrame},
                 [](SyntheticSection *SS) { SS->finalizeContents(); });

  for (Symbol *S : Symtab->getSymbols())
    S->body()->IsPreemptible |= computeIsPreemptible(*S->body());

  // Scan relocations. This must be done after every symbol is declared so that
  // we can correctly decide if a dynamic relocation is needed.
  if (!Config->Relocatable)
    forEachRelSec(scanRelocations<ELFT>);

  if (InX::Plt && !InX::Plt->empty())
    InX::Plt->addSymbols();
  if (InX::Iplt && !InX::Iplt->empty())
    InX::Iplt->addSymbols();

  // Now that we have defined all possible global symbols including linker-
  // synthesized ones. Visit all symbols to give the finishing touches.
  for (Symbol *S : Symtab->getSymbols()) {
    SymbolBody *Body = S->body();

    if (!includeInSymtab(*Body))
      continue;
    if (InX::SymTab)
      InX::SymTab->addSymbol(Body);

    if (InX::DynSymTab && S->includeInDynsym()) {
      InX::DynSymTab->addSymbol(Body);
      if (auto *SS = dyn_cast<SharedSymbol>(Body))
        if (cast<SharedFile<ELFT>>(S->File)->isNeeded())
          In<ELFT>::VerNeed->addSymbol(SS);
    }
  }

  // Do not proceed if there was an undefined symbol.
  if (errorCount())
    return;

  addPredefinedSections();
  removeUnusedSyntheticSections();

  sortSections();
  Script->removeEmptyCommands();

  // Now that we have the final list, create a list of all the
  // OutputSections for convenience.
  for (BaseCommand *Base : Script->SectionCommands)
    if (auto *Sec = dyn_cast<OutputSection>(Base))
      OutputSections.push_back(Sec);

  // Prefer command line supplied address over other constraints.
  for (OutputSection *Sec : OutputSections) {
    auto I = Config->SectionStartMap.find(Sec->Name);
    if (I != Config->SectionStartMap.end())
      Sec->AddrExpr = [=] { return I->second; };
  }

  // This is a bit of a hack. A value of 0 means undef, so we set it
  // to 1 t make __ehdr_start defined. The section number is not
  // particularly relevant.
  Out::ElfHeader->SectionIndex = 1;

  unsigned I = 1;
  for (OutputSection *Sec : OutputSections) {
    Sec->SectionIndex = I++;
    Sec->ShName = InX::ShStrTab->addString(Sec->Name);
  }

  // Binary and relocatable output does not have PHDRS.
  // The headers have to be created before finalize as that can influence the
  // image base and the dynamic section on mips includes the image base.
  if (!Config->Relocatable && !Config->OFormatBinary) {
    Phdrs = Script->hasPhdrsCommands() ? Script->createPhdrs() : createPhdrs();
    addPtArmExid(Phdrs);
    Out::ProgramHeaders->Size = sizeof(Elf_Phdr) * Phdrs.size();
  }

  // Some symbols are defined in term of program headers. Now that we
  // have the headers, we can find out which sections they point to.
  setReservedSymbolSections();

  // Dynamic section must be the last one in this list and dynamic
  // symbol table section (DynSymTab) must be the first one.
  applySynthetic({InX::DynSymTab,     InX::Bss,          InX::BssRelRo,
                  InX::GnuHashTab,    InX::HashTab,      InX::SymTab,
                  InX::ShStrTab,      InX::StrTab,       In<ELFT>::VerDef,
                  InX::DynStrTab,     InX::Got,          InX::MipsGot,
                  InX::IgotPlt,       InX::GotPlt,       In<ELFT>::RelaDyn,
                  In<ELFT>::RelaIplt, In<ELFT>::RelaPlt, InX::Plt,
                  InX::Iplt,          InX::EhFrameHdr,   In<ELFT>::VerSym,
                  In<ELFT>::VerNeed,  InX::Dynamic},
                 [](SyntheticSection *SS) { SS->finalizeContents(); });

  if (!Script->HasSectionsCommand && !Config->Relocatable)
    fixSectionAlignments();

  // Some architectures use small displacements for jump instructions.
  // It is linker's responsibility to create thunks containing long
  // jump instructions if jump targets are too far. Create thunks.
  if (Target->NeedsThunks || Config->AndroidPackDynRelocs) {
    ThunkCreator TC;
    bool Changed;
    do {
      Script->assignAddresses();
      Changed = false;
      if (Target->NeedsThunks)
        Changed |= TC.createThunks(OutputSections);
      if (InX::MipsGot)
        InX::MipsGot->updateAllocSize();
      Changed |= In<ELFT>::RelaDyn->updateAllocSize();
    } while (Changed);
  }

  // Fill other section headers. The dynamic table is finalized
  // at the end because some tags like RELSZ depend on result
  // of finalizing other sections.
  for (OutputSection *Sec : OutputSections)
    Sec->finalize<ELFT>();

  // createThunks may have added local symbols to the static symbol table
  applySynthetic({InX::SymTab, InX::ShStrTab, InX::StrTab},
                 [](SyntheticSection *SS) { SS->postThunkContents(); });
}

template <class ELFT> void Writer<ELFT>::addPredefinedSections() {
  // ARM ABI requires .ARM.exidx to be terminated by some piece of data.
  // We have the terminater synthetic section class. Add that at the end.
  OutputSection *Cmd = findSection(".ARM.exidx");
  if (!Cmd || !Cmd->Live || Config->Relocatable)
    return;

  auto *Sentinel = make<ARMExidxSentinelSection>();
  Cmd->addSection(Sentinel);
}

// The linker is expected to define SECNAME_start and SECNAME_end
// symbols for a few sections. This function defines them.
template <class ELFT> void Writer<ELFT>::addStartEndSymbols() {
  auto Define = [&](StringRef Start, StringRef End, OutputSection *OS) {
    // These symbols resolve to the image base if the section does not exist.
    // A special value -1 indicates end of the section.
    if (OS) {
      addOptionalRegular<ELFT>(Start, OS, 0);
      addOptionalRegular<ELFT>(End, OS, -1);
    } else {
      if (Config->Pic)
        OS = Out::ElfHeader;
      addOptionalRegular<ELFT>(Start, OS, 0);
      addOptionalRegular<ELFT>(End, OS, 0);
    }
  };

  Define("__preinit_array_start", "__preinit_array_end", Out::PreinitArray);
  Define("__init_array_start", "__init_array_end", Out::InitArray);
  Define("__fini_array_start", "__fini_array_end", Out::FiniArray);

  if (OutputSection *Sec = findSection(".ARM.exidx"))
    Define("__exidx_start", "__exidx_end", Sec);
}

// If a section name is valid as a C identifier (which is rare because of
// the leading '.'), linkers are expected to define __start_<secname> and
// __stop_<secname> symbols. They are at beginning and end of the section,
// respectively. This is not requested by the ELF standard, but GNU ld and
// gold provide the feature, and used by many programs.
template <class ELFT>
void Writer<ELFT>::addStartStopSymbols(OutputSection *Sec) {
  StringRef S = Sec->Name;
  if (!isValidCIdentifier(S))
    return;
  addOptionalRegular<ELFT>(Saver.save("__start_" + S), Sec, 0, STV_DEFAULT);
  addOptionalRegular<ELFT>(Saver.save("__stop_" + S), Sec, -1, STV_DEFAULT);
}

template <class ELFT> OutputSection *Writer<ELFT>::findSection(StringRef Name) {
  for (BaseCommand *Base : Script->SectionCommands)
    if (auto *Sec = dyn_cast<OutputSection>(Base))
      if (Sec->Name == Name)
        return Sec;
  return nullptr;
}

static bool needsPtLoad(OutputSection *Sec) {
  if (!(Sec->Flags & SHF_ALLOC))
    return false;

  // Don't allocate VA space for TLS NOBITS sections. The PT_TLS PHDR is
  // responsible for allocating space for them, not the PT_LOAD that
  // contains the TLS initialization image.
  if (Sec->Flags & SHF_TLS && Sec->Type == SHT_NOBITS)
    return false;
  return true;
}

// Linker scripts are responsible for aligning addresses. Unfortunately, most
// linker scripts are designed for creating two PT_LOADs only, one RX and one
// RW. This means that there is no alignment in the RO to RX transition and we
// cannot create a PT_LOAD there.
static uint64_t computeFlags(uint64_t Flags) {
  if (Config->Omagic)
    return PF_R | PF_W | PF_X;
  if (Config->SingleRoRx && !(Flags & PF_W))
    return Flags | PF_X;
  return Flags;
}

// Decide which program headers to create and which sections to include in each
// one.
template <class ELFT> std::vector<PhdrEntry *> Writer<ELFT>::createPhdrs() {
  std::vector<PhdrEntry *> Ret;
  auto AddHdr = [&](unsigned Type, unsigned Flags) -> PhdrEntry * {
    Ret.push_back(make<PhdrEntry>(Type, Flags));
    return Ret.back();
  };

  // The first phdr entry is PT_PHDR which describes the program header itself.
  AddHdr(PT_PHDR, PF_R)->add(Out::ProgramHeaders);

  // PT_INTERP must be the second entry if exists.
  if (OutputSection *Cmd = findSection(".interp"))
    AddHdr(PT_INTERP, Cmd->getPhdrFlags())->add(Cmd);

  // Add the first PT_LOAD segment for regular output sections.
  uint64_t Flags = computeFlags(PF_R);
  PhdrEntry *Load = AddHdr(PT_LOAD, Flags);

  // Add the headers. We will remove them if they don't fit.
  Load->add(Out::ElfHeader);
  Load->add(Out::ProgramHeaders);

  for (OutputSection *Sec : OutputSections) {
    if (!(Sec->Flags & SHF_ALLOC))
      break;
    if (!needsPtLoad(Sec))
      continue;

    // Segments are contiguous memory regions that has the same attributes
    // (e.g. executable or writable). There is one phdr for each segment.
    // Therefore, we need to create a new phdr when the next section has
    // different flags or is loaded at a discontiguous address using AT linker
    // script command.
    uint64_t NewFlags = computeFlags(Sec->getPhdrFlags());
    if (Sec->LMAExpr || Flags != NewFlags) {
      Load = AddHdr(PT_LOAD, NewFlags);
      Flags = NewFlags;
    }

    Load->add(Sec);
  }

  // Add a TLS segment if any.
  PhdrEntry *TlsHdr = make<PhdrEntry>(PT_TLS, PF_R);
  for (OutputSection *Sec : OutputSections)
    if (Sec->Flags & SHF_TLS)
      TlsHdr->add(Sec);
  if (TlsHdr->FirstSec)
    Ret.push_back(TlsHdr);

  // Add an entry for .dynamic.
  if (InX::DynSymTab)
    AddHdr(PT_DYNAMIC, InX::Dynamic->getParent()->getPhdrFlags())
        ->add(InX::Dynamic->getParent());

  // PT_GNU_RELRO includes all sections that should be marked as
  // read-only by dynamic linker after proccessing relocations.
  PhdrEntry *RelRo = make<PhdrEntry>(PT_GNU_RELRO, PF_R);
  for (OutputSection *Sec : OutputSections)
    if (needsPtLoad(Sec) && isRelroSection(Sec))
      RelRo->add(Sec);
  if (RelRo->FirstSec)
    Ret.push_back(RelRo);

  // PT_GNU_EH_FRAME is a special section pointing on .eh_frame_hdr.
  if (!InX::EhFrame->empty() && InX::EhFrameHdr && InX::EhFrame->getParent() &&
      InX::EhFrameHdr->getParent())
    AddHdr(PT_GNU_EH_FRAME, InX::EhFrameHdr->getParent()->getPhdrFlags())
        ->add(InX::EhFrameHdr->getParent());

  // PT_OPENBSD_RANDOMIZE is an OpenBSD-specific feature. That makes
  // the dynamic linker fill the segment with random data.
  if (OutputSection *Cmd = findSection(".openbsd.randomdata"))
    AddHdr(PT_OPENBSD_RANDOMIZE, Cmd->getPhdrFlags())->add(Cmd);

  // PT_GNU_STACK is a special section to tell the loader to make the
  // pages for the stack non-executable. If you really want an executable
  // stack, you can pass -z execstack, but that's not recommended for
  // security reasons.
  unsigned Perm;
  if (Config->ZExecstack)
    Perm = PF_R | PF_W | PF_X;
  else
    Perm = PF_R | PF_W;
  AddHdr(PT_GNU_STACK, Perm)->p_memsz = Config->ZStackSize;

  // PT_OPENBSD_WXNEEDED is a OpenBSD-specific header to mark the executable
  // is expected to perform W^X violations, such as calling mprotect(2) or
  // mmap(2) with PROT_WRITE | PROT_EXEC, which is prohibited by default on
  // OpenBSD.
  if (Config->ZWxneeded)
    AddHdr(PT_OPENBSD_WXNEEDED, PF_X);

  // Create one PT_NOTE per a group of contiguous .note sections.
  PhdrEntry *Note = nullptr;
  for (OutputSection *Sec : OutputSections) {
    if (Sec->Type == SHT_NOTE) {
      if (!Note || Sec->LMAExpr)
        Note = AddHdr(PT_NOTE, PF_R);
      Note->add(Sec);
    } else {
      Note = nullptr;
    }
  }
  return Ret;
}

template <class ELFT>
void Writer<ELFT>::addPtArmExid(std::vector<PhdrEntry *> &Phdrs) {
  if (Config->EMachine != EM_ARM)
    return;
  auto I = llvm::find_if(OutputSections, [](OutputSection *Cmd) {
    return Cmd->Type == SHT_ARM_EXIDX;
  });
  if (I == OutputSections.end())
    return;

  // PT_ARM_EXIDX is the ARM EHABI equivalent of PT_GNU_EH_FRAME
  PhdrEntry *ARMExidx = make<PhdrEntry>(PT_ARM_EXIDX, PF_R);
  ARMExidx->add(*I);
  Phdrs.push_back(ARMExidx);
}

// The first section of each PT_LOAD, the first section in PT_GNU_RELRO and the
// first section after PT_GNU_RELRO have to be page aligned so that the dynamic
// linker can set the permissions.
template <class ELFT> void Writer<ELFT>::fixSectionAlignments() {
  auto PageAlign = [](OutputSection *Cmd) {
    if (Cmd && !Cmd->AddrExpr)
      Cmd->AddrExpr = [=] {
        return alignTo(Script->getDot(), Config->MaxPageSize);
      };
  };

  for (const PhdrEntry *P : Phdrs)
    if (P->p_type == PT_LOAD && P->FirstSec)
      PageAlign(P->FirstSec);

  for (const PhdrEntry *P : Phdrs) {
    if (P->p_type != PT_GNU_RELRO)
      continue;
    if (P->FirstSec)
      PageAlign(P->FirstSec);
    // Find the first section after PT_GNU_RELRO. If it is in a PT_LOAD we
    // have to align it to a page.
    auto End = OutputSections.end();
    auto I = std::find(OutputSections.begin(), End, P->LastSec);
    if (I == End || (I + 1) == End)
      continue;
    OutputSection *Cmd = (*(I + 1));
    if (needsPtLoad(Cmd))
      PageAlign(Cmd);
  }
}

// Adjusts the file alignment for a given output section and returns
// its new file offset. The file offset must be the same with its
// virtual address (modulo the page size) so that the loader can load
// executables without any address adjustment.
static uint64_t getFileAlignment(uint64_t Off, OutputSection *Cmd) {
  // If the section is not in a PT_LOAD, we just have to align it.
  if (!Cmd->PtLoad)
    return alignTo(Off, Cmd->Alignment);

  OutputSection *First = Cmd->PtLoad->FirstSec;
  // The first section in a PT_LOAD has to have congruent offset and address
  // module the page size.
  if (Cmd == First)
    return alignTo(Off, std::max<uint64_t>(Cmd->Alignment, Config->MaxPageSize),
                   Cmd->Addr);

  // If two sections share the same PT_LOAD the file offset is calculated
  // using this formula: Off2 = Off1 + (VA2 - VA1).
  return First->Offset + Cmd->Addr - First->Addr;
}

static uint64_t setOffset(OutputSection *Cmd, uint64_t Off) {
  if (Cmd->Type == SHT_NOBITS) {
    Cmd->Offset = Off;
    return Off;
  }

  Off = getFileAlignment(Off, Cmd);
  Cmd->Offset = Off;
  return Off + Cmd->Size;
}

template <class ELFT> void Writer<ELFT>::assignFileOffsetsBinary() {
  uint64_t Off = 0;
  for (OutputSection *Sec : OutputSections)
    if (Sec->Flags & SHF_ALLOC)
      Off = setOffset(Sec, Off);
  FileSize = alignTo(Off, Config->Wordsize);
}

// Assign file offsets to output sections.
template <class ELFT> void Writer<ELFT>::assignFileOffsets() {
  uint64_t Off = 0;
  Off = setOffset(Out::ElfHeader, Off);
  Off = setOffset(Out::ProgramHeaders, Off);

  PhdrEntry *LastRX = nullptr;
  for (PhdrEntry *P : Phdrs)
    if (P->p_type == PT_LOAD && (P->p_flags & PF_X))
      LastRX = P;

  for (OutputSection *Sec : OutputSections) {
    Off = setOffset(Sec, Off);
    if (Script->HasSectionsCommand)
      continue;
    // If this is a last section of the last executable segment and that
    // segment is the last loadable segment, align the offset of the
    // following section to avoid loading non-segments parts of the file.
    if (LastRX && LastRX->LastSec == Sec)
      Off = alignTo(Off, Target->PageSize);
  }

  SectionHeaderOff = alignTo(Off, Config->Wordsize);
  FileSize = SectionHeaderOff + (OutputSections.size() + 1) * sizeof(Elf_Shdr);
}

// Finalize the program headers. We call this function after we assign
// file offsets and VAs to all sections.
template <class ELFT> void Writer<ELFT>::setPhdrs() {
  for (PhdrEntry *P : Phdrs) {
    OutputSection *First = P->FirstSec;
    OutputSection *Last = P->LastSec;
    if (First) {
      P->p_filesz = Last->Offset - First->Offset;
      if (Last->Type != SHT_NOBITS)
        P->p_filesz += Last->Size;
      P->p_memsz = Last->Addr + Last->Size - First->Addr;
      P->p_offset = First->Offset;
      P->p_vaddr = First->Addr;
      if (!P->HasLMA)
        P->p_paddr = First->getLMA();
    }
    if (P->p_type == PT_LOAD)
      P->p_align = std::max<uint64_t>(P->p_align, Config->MaxPageSize);
    else if (P->p_type == PT_GNU_RELRO) {
      P->p_align = 1;
      // The glibc dynamic loader rounds the size down, so we need to round up
      // to protect the last page. This is a no-op on FreeBSD which always
      // rounds up.
      P->p_memsz = alignTo(P->p_memsz, Target->PageSize);
    }

    // The TLS pointer goes after PT_TLS. At least glibc will align it,
    // so round up the size to make sure the offsets are correct.
    if (P->p_type == PT_TLS) {
      Out::TlsPhdr = P;
      if (P->p_memsz)
        P->p_memsz = alignTo(P->p_memsz, P->p_align);
    }
  }
}

// The entry point address is chosen in the following ways.
//
// 1. the '-e' entry command-line option;
// 2. the ENTRY(symbol) command in a linker control script;
// 3. the value of the symbol start, if present;
// 4. the number represented by the entry symbol, if it is a number;
// 5. the address of the first byte of the .text section, if present;
// 6. the address 0.
template <class ELFT> uint64_t Writer<ELFT>::getEntryAddr() {
  // Case 1, 2 or 3
  if (SymbolBody *B = Symtab->find(Config->Entry))
    return B->getVA();

  // Case 4
  uint64_t Addr;
  if (to_integer(Config->Entry, Addr))
    return Addr;

  // Case 5
  if (OutputSection *Sec = findSection(".text")) {
    if (Config->WarnMissingEntry)
      warn("cannot find entry symbol " + Config->Entry + "; defaulting to 0x" +
           utohexstr(Sec->Addr));
    return Sec->Addr;
  }

  // Case 6
  if (Config->WarnMissingEntry)
    warn("cannot find entry symbol " + Config->Entry +
         "; not setting start address");
  return 0;
}

static uint16_t getELFType() {
  if (Config->Pic)
    return ET_DYN;
  if (Config->Relocatable)
    return ET_REL;
  return ET_EXEC;
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  memcpy(Buf, "\177ELF", 4);

  // Write the ELF header.
  auto *EHdr = reinterpret_cast<Elf_Ehdr *>(Buf);
  EHdr->e_ident[EI_CLASS] = Config->Is64 ? ELFCLASS64 : ELFCLASS32;
  EHdr->e_ident[EI_DATA] = Config->IsLE ? ELFDATA2LSB : ELFDATA2MSB;
  EHdr->e_ident[EI_VERSION] = EV_CURRENT;
  EHdr->e_ident[EI_OSABI] = Config->OSABI;
  EHdr->e_type = getELFType();
  EHdr->e_machine = Config->EMachine;
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = getEntryAddr();
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_flags = Config->EFlags;
  EHdr->e_ehsize = sizeof(Elf_Ehdr);
  EHdr->e_phnum = Phdrs.size();
  EHdr->e_shentsize = sizeof(Elf_Shdr);
  EHdr->e_shnum = OutputSections.size() + 1;
  EHdr->e_shstrndx = InX::ShStrTab->getParent()->SectionIndex;

  if (!Config->Relocatable) {
    EHdr->e_phoff = sizeof(Elf_Ehdr);
    EHdr->e_phentsize = sizeof(Elf_Phdr);
  }

  // Write the program header table.
  auto *HBuf = reinterpret_cast<Elf_Phdr *>(Buf + EHdr->e_phoff);
  for (PhdrEntry *P : Phdrs) {
    HBuf->p_type = P->p_type;
    HBuf->p_flags = P->p_flags;
    HBuf->p_offset = P->p_offset;
    HBuf->p_vaddr = P->p_vaddr;
    HBuf->p_paddr = P->p_paddr;
    HBuf->p_filesz = P->p_filesz;
    HBuf->p_memsz = P->p_memsz;
    HBuf->p_align = P->p_align;
    ++HBuf;
  }

  // Write the section header table. Note that the first table entry is null.
  auto *SHdrs = reinterpret_cast<Elf_Shdr *>(Buf + EHdr->e_shoff);
  for (OutputSection *Sec : OutputSections)
    Sec->writeHeaderTo<ELFT>(++SHdrs);
}

// Open a result file.
template <class ELFT> void Writer<ELFT>::openFile() {
  if (!Config->Is64 && FileSize > UINT32_MAX) {
    error("output file too large: " + Twine(FileSize) + " bytes");
    return;
  }

  unlinkAsync(Config->OutputFile);
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Config->OutputFile, FileSize,
                               FileOutputBuffer::F_executable);

  if (auto EC = BufferOrErr.getError())
    error("failed to open " + Config->OutputFile + ": " + EC.message());
  else
    Buffer = std::move(*BufferOrErr);
}

template <class ELFT> void Writer<ELFT>::writeSectionsBinary() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSection *Sec : OutputSections)
    if (Sec->Flags & SHF_ALLOC)
      Sec->writeTo<ELFT>(Buf + Sec->Offset);
}

static void fillTrap(uint8_t *I, uint8_t *End) {
  for (; I + 4 <= End; I += 4)
    memcpy(I, &Target->TrapInstr, 4);
}

// Fill the last page of executable segments with trap instructions
// instead of leaving them as zero. Even though it is not required by any
// standard, it is in general a good thing to do for security reasons.
//
// We'll leave other pages in segments as-is because the rest will be
// overwritten by output sections.
template <class ELFT> void Writer<ELFT>::writeTrapInstr() {
  if (Script->HasSectionsCommand)
    return;

  // Fill the last page.
  uint8_t *Buf = Buffer->getBufferStart();
  for (PhdrEntry *P : Phdrs)
    if (P->p_type == PT_LOAD && (P->p_flags & PF_X))
      fillTrap(Buf + alignDown(P->p_offset + P->p_filesz, Target->PageSize),
               Buf + alignTo(P->p_offset + P->p_filesz, Target->PageSize));

  // Round up the file size of the last segment to the page boundary iff it is
  // an executable segment to ensure that other tools don't accidentally
  // trim the instruction padding (e.g. when stripping the file).
  PhdrEntry *Last = nullptr;
  for (PhdrEntry *P : Phdrs)
    if (P->p_type == PT_LOAD)
      Last = P;

  if (Last && (Last->p_flags & PF_X))
    Last->p_memsz = Last->p_filesz = alignTo(Last->p_filesz, Target->PageSize);
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();

  // PPC64 needs to process relocations in the .opd section
  // before processing relocations in code-containing sections.
  if (auto *OpdCmd = findSection(".opd")) {
    Out::Opd = OpdCmd;
    Out::OpdBuf = Buf + Out::Opd->Offset;
    OpdCmd->template writeTo<ELFT>(Buf + Out::Opd->Offset);
  }

  OutputSection *EhFrameHdr = nullptr;
  if (InX::EhFrameHdr && !InX::EhFrameHdr->empty())
    EhFrameHdr = InX::EhFrameHdr->getParent();

  // In -r or -emit-relocs mode, write the relocation sections first as in
  // ELf_Rel targets we might find out that we need to modify the relocated
  // section while doing it.
  for (OutputSection *Sec : OutputSections)
    if (Sec->Type == SHT_REL || Sec->Type == SHT_RELA)
      Sec->writeTo<ELFT>(Buf + Sec->Offset);

  for (OutputSection *Sec : OutputSections)
    if (Sec != Out::Opd && Sec != EhFrameHdr && Sec->Type != SHT_REL &&
        Sec->Type != SHT_RELA)
      Sec->writeTo<ELFT>(Buf + Sec->Offset);

  // The .eh_frame_hdr depends on .eh_frame section contents, therefore
  // it should be written after .eh_frame is written.
  if (EhFrameHdr)
    EhFrameHdr->writeTo<ELFT>(Buf + EhFrameHdr->Offset);
}

template <class ELFT> void Writer<ELFT>::writeBuildId() {
  if (!InX::BuildId || !InX::BuildId->getParent())
    return;

  // Compute a hash of all sections of the output file.
  uint8_t *Start = Buffer->getBufferStart();
  uint8_t *End = Start + FileSize;
  InX::BuildId->writeBuildId({Start, End});
}

template void elf::writeResult<ELF32LE>();
template void elf::writeResult<ELF32BE>();
template void elf::writeResult<ELF64LE>();
template void elf::writeResult<ELF64BE>();
