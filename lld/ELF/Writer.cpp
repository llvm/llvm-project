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
#include "LinkerScript.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Relocations.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <thread>

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
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::SymRange Elf_Sym_Range;
  typedef typename ELFT::Rela Elf_Rela;
  void run();

private:
  typedef PhdrEntry<ELFT> Phdr;

  void createSyntheticSections();
  void copyLocalSymbols();
  void addReservedSymbols();
  void addInputSec(InputSectionBase<ELFT> *S);
  void createSections();
  void forEachRelSec(std::function<void(InputSectionBase<ELFT> &)> Fn);
  void sortSections();
  void finalizeSections();
  void addPredefinedSections();

  std::vector<Phdr> createPhdrs();
  void removeEmptyPTLoad();
  void addPtArmExid(std::vector<Phdr> &Phdrs);
  void assignAddresses();
  void assignFileOffsets();
  void assignFileOffsetsBinary();
  void setPhdrs();
  void fixHeaders();
  void fixSectionAlignments();
  void fixAbsoluteSymbols();
  void openFile();
  void writeHeader();
  void writeSections();
  void writeSectionsBinary();
  void writeBuildId();

  std::unique_ptr<FileOutputBuffer> Buffer;

  std::vector<OutputSectionBase *> OutputSections;
  OutputSectionFactory<ELFT> Factory;

  void addRelIpltSymbols();
  void addStartEndSymbols();
  void addStartStopSymbols(OutputSectionBase *Sec);
  uintX_t getEntryAddr();
  OutputSectionBase *findSection(StringRef Name);

  std::vector<Phdr> Phdrs;

  uintX_t FileSize;
  uintX_t SectionHeaderOff;
};
} // anonymous namespace

StringRef elf::getOutputSectionName(StringRef Name) {
  if (Config->Relocatable)
    return Name;

  for (StringRef V :
       {".text.", ".rodata.", ".data.rel.ro.", ".data.", ".bss.",
        ".init_array.", ".fini_array.", ".ctors.", ".dtors.", ".tbss.",
        ".gcc_except_table.", ".tdata.", ".ARM.exidx."}) {
    StringRef Prefix = V.drop_back();
    if (Name.startswith(V) || Name == Prefix)
      return Prefix;
  }

  // CommonSection is identified as "COMMON" in linker scripts.
  // By default, it should go to .bss section.
  if (Name == "COMMON")
    return ".bss";

  // ".zdebug_" is a prefix for ZLIB-compressed sections.
  // Because we decompressed input sections, we want to remove 'z'.
  if (Name.startswith(".zdebug_"))
    return Saver.save(Twine(".") + Name.substr(2));
  return Name;
}

template <class ELFT> void elf::reportDiscarded(InputSectionBase<ELFT> *IS) {
  if (!Config->PrintGcSections)
    return;
  errs() << "removing unused section from '" << IS->Name << "' in file '"
         << IS->getFile()->getName() << "'\n";
}

template <class ELFT> static bool needsInterpSection() {
  return !Symtab<ELFT>::X->getSharedFiles().empty() &&
         !Config->DynamicLinker.empty() &&
         !Script<ELFT>::X->ignoreInterpSection();
}

template <class ELFT> void elf::writeResult() { Writer<ELFT>().run(); }

template <class ELFT> void Writer<ELFT>::removeEmptyPTLoad() {
  auto I = std::remove_if(Phdrs.begin(), Phdrs.end(), [&](const Phdr &P) {
    if (P.H.p_type != PT_LOAD)
      return false;
    if (!P.First)
      return true;
    uintX_t Size = P.Last->Addr + P.Last->Size - P.First->Addr;
    return Size == 0;
  });
  Phdrs.erase(I, Phdrs.end());
}

// The main function of the writer.
template <class ELFT> void Writer<ELFT>::run() {
  // Create linker-synthesized sections such as .got or .plt.
  // Such sections are of type input section.
  createSyntheticSections();

  // We need to create some reserved symbols such as _end. Create them.
  if (!Config->Relocatable)
    addReservedSymbols();

  // Some architectures use small displacements for jump instructions.
  // It is linker's responsibility to create thunks containing long
  // jump instructions if jump targets are too far. Create thunks.
  if (Target->NeedsThunks)
    forEachRelSec(createThunks<ELFT>);

  // Create output sections.
  Script<ELFT>::X->OutputSections = &OutputSections;
  if (ScriptConfig->HasSections) {
    // If linker script contains SECTIONS commands, let it create sections.
    Script<ELFT>::X->processCommands(Factory);

    // Linker scripts may have left some input sections unassigned.
    // Assign such sections using the default rule.
    Script<ELFT>::X->addOrphanSections(Factory);
  } else {
    // If linker script does not contain SECTIONS commands, create
    // output sections by default rules. We still need to give the
    // linker script a chance to run, because it might contain
    // non-SECTIONS commands such as ASSERT.
    createSections();
    Script<ELFT>::X->processCommands(Factory);
  }

  if (Config->Discard != DiscardPolicy::All)
    copyLocalSymbols();

  // Now that we have a complete set of output sections. This function
  // completes section contents. For example, we need to add strings
  // to the string table, and add entries to .got and .plt.
  // finalizeSections does that.
  finalizeSections();
  if (ErrorCount)
    return;

  if (Config->Relocatable) {
    assignFileOffsets();
  } else {
    // Binary output does not have PHDRS.
    if (!Config->OFormatBinary) {
      Phdrs = Script<ELFT>::X->hasPhdrsCommands()
                  ? Script<ELFT>::X->createPhdrs()
                  : createPhdrs();
      addPtArmExid(Phdrs);
      fixHeaders();
    }

    if (ScriptConfig->HasSections) {
      Script<ELFT>::X->assignAddresses(Phdrs);
    } else {
      fixSectionAlignments();
      assignAddresses();
    }

    // Remove empty PT_LOAD to avoid causing the dynamic linker to try to mmap a
    // 0 sized region. This has to be done late since only after assignAddresses
    // we know the size of the sections.
    removeEmptyPTLoad();

    if (!Config->OFormatBinary)
      assignFileOffsets();
    else
      assignFileOffsetsBinary();

    setPhdrs();
    fixAbsoluteSymbols();
  }

  // Write the result down to a file.
  openFile();
  if (ErrorCount)
    return;
  if (!Config->OFormatBinary) {
    writeHeader();
    writeSections();
  } else {
    writeSectionsBinary();
  }

  // Backfill .note.gnu.build-id section content. This is done at last
  // because the content is usually a hash value of the entire output file.
  writeBuildId();
  if (ErrorCount)
    return;

  if (auto EC = Buffer->commit())
    error(EC, "failed to write to the output file");

  // Flush the output streams and exit immediately. A full shutdown
  // is a good test that we are keeping track of all allocated memory,
  // but actually freeing it is a waste of time in a regular linker run.
  if (Config->ExitEarly)
    exitLld(0);
}

// Initialize Out<ELFT> members.
template <class ELFT> void Writer<ELFT>::createSyntheticSections() {
  // Initialize all pointers with NULL. This is needed because
  // you can call lld::elf::main more than once as a library.
  memset(&Out<ELFT>::First, 0, sizeof(Out<ELFT>));

  // Create singleton output sections.
  Out<ELFT>::Bss =
      make<OutputSection<ELFT>>(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE);
  In<ELFT>::DynStrTab = make<StringTableSection<ELFT>>(".dynstr", true);
  In<ELFT>::Dynamic = make<DynamicSection<ELFT>>();
  Out<ELFT>::EhFrame = make<EhOutputSection<ELFT>>();
  In<ELFT>::RelaDyn = make<RelocationSection<ELFT>>(
      Config->Rela ? ".rela.dyn" : ".rel.dyn", Config->ZCombreloc);
  In<ELFT>::ShStrTab = make<StringTableSection<ELFT>>(".shstrtab", false);

  Out<ELFT>::ElfHeader = make<OutputSectionBase>("", 0, SHF_ALLOC);
  Out<ELFT>::ElfHeader->Size = sizeof(Elf_Ehdr);
  Out<ELFT>::ProgramHeaders = make<OutputSectionBase>("", 0, SHF_ALLOC);
  Out<ELFT>::ProgramHeaders->updateAlignment(sizeof(uintX_t));

  if (needsInterpSection<ELFT>()) {
    In<ELFT>::Interp = createInterpSection<ELFT>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::Interp);
  } else {
    In<ELFT>::Interp = nullptr;
  }

  if (!Config->Relocatable)
    Symtab<ELFT>::X->Sections.push_back(createCommentSection<ELFT>());

  if (Config->Strip != StripPolicy::All) {
    In<ELFT>::StrTab = make<StringTableSection<ELFT>>(".strtab", false);
    In<ELFT>::SymTab = make<SymbolTableSection<ELFT>>(*In<ELFT>::StrTab);
  }

  if (Config->BuildId != BuildIdKind::None) {
    In<ELFT>::BuildId = make<BuildIdSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::BuildId);
  }

  InputSection<ELFT> *Common = createCommonSection<ELFT>();
  if (!Common->Data.empty()) {
    In<ELFT>::Common = Common;
    Symtab<ELFT>::X->Sections.push_back(Common);
  }

  // Add MIPS-specific sections.
  bool HasDynSymTab = !Symtab<ELFT>::X->getSharedFiles().empty() || Config->Pic;
  if (Config->EMachine == EM_MIPS) {
    if (!Config->Shared && HasDynSymTab) {
      In<ELFT>::MipsRldMap = make<MipsRldMapSection<ELFT>>();
      Symtab<ELFT>::X->Sections.push_back(In<ELFT>::MipsRldMap);
    }
    if (auto *Sec = MipsAbiFlagsSection<ELFT>::create())
      Symtab<ELFT>::X->Sections.push_back(Sec);
    if (auto *Sec = MipsOptionsSection<ELFT>::create())
      Symtab<ELFT>::X->Sections.push_back(Sec);
    if (auto *Sec = MipsReginfoSection<ELFT>::create())
      Symtab<ELFT>::X->Sections.push_back(Sec);
  }

  if (HasDynSymTab) {
    In<ELFT>::DynSymTab = make<SymbolTableSection<ELFT>>(*In<ELFT>::DynStrTab);
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::DynSymTab);

    In<ELFT>::VerSym = make<VersionTableSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::VerSym);

    if (!Config->VersionDefinitions.empty()) {
      In<ELFT>::VerDef = make<VersionDefinitionSection<ELFT>>();
      Symtab<ELFT>::X->Sections.push_back(In<ELFT>::VerDef);
    }

    In<ELFT>::VerNeed = make<VersionNeedSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::VerNeed);

    if (Config->GnuHash) {
      In<ELFT>::GnuHashTab = make<GnuHashTableSection<ELFT>>();
      Symtab<ELFT>::X->Sections.push_back(In<ELFT>::GnuHashTab);
    }

    if (Config->SysvHash) {
      In<ELFT>::HashTab = make<HashTableSection<ELFT>>();
      Symtab<ELFT>::X->Sections.push_back(In<ELFT>::HashTab);
    }

    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::Dynamic);
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::DynStrTab);
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::RelaDyn);
  }

  // Add .got. MIPS' .got is so different from the other archs,
  // it has its own class.
  if (Config->EMachine == EM_MIPS) {
    In<ELFT>::MipsGot = make<MipsGotSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::MipsGot);
  } else {
    In<ELFT>::Got = make<GotSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::Got);
  }

  In<ELFT>::GotPlt = make<GotPltSection<ELFT>>();
  Symtab<ELFT>::X->Sections.push_back(In<ELFT>::GotPlt);

  if (Config->GdbIndex) {
    In<ELFT>::GdbIndex = make<GdbIndexSection<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::GdbIndex);
  }

  // We always need to add rel[a].plt to output if it has entries.
  // Even for static linking it can contain R_[*]_IRELATIVE relocations.
  In<ELFT>::RelaPlt = make<RelocationSection<ELFT>>(
      Config->Rela ? ".rela.plt" : ".rel.plt", false /*Sort*/);
  Symtab<ELFT>::X->Sections.push_back(In<ELFT>::RelaPlt);

  In<ELFT>::Plt = make<PltSection<ELFT>>();
  Symtab<ELFT>::X->Sections.push_back(In<ELFT>::Plt);

  if (Config->EhFrameHdr) {
    In<ELFT>::EhFrameHdr = make<EhFrameHeader<ELFT>>();
    Symtab<ELFT>::X->Sections.push_back(In<ELFT>::EhFrameHdr);
  }
}

template <class ELFT>
static bool shouldKeepInSymtab(InputSectionBase<ELFT> *Sec, StringRef SymName,
                               const SymbolBody &B) {
  if (B.isFile())
    return false;

  // We keep sections in symtab for relocatable output.
  if (B.isSection())
    return Config->Relocatable;

  // If sym references a section in a discarded group, don't keep it.
  if (Sec == &InputSection<ELFT>::Discarded)
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

template <class ELFT> static bool includeInSymtab(const SymbolBody &B) {
  if (!B.isLocal() && !B.symbol()->IsUsedInRegularObj)
    return false;

  if (auto *D = dyn_cast<DefinedRegular<ELFT>>(&B)) {
    // Always include absolute symbols.
    if (!D->Section)
      return true;
    // Exclude symbols pointing to garbage-collected sections.
    if (!D->Section->Live)
      return false;
    if (auto *S = dyn_cast<MergeInputSection<ELFT>>(D->Section))
      if (!S->getSectionPiece(D->Value)->Live)
        return false;
  }
  return true;
}

// Local symbols are not in the linker's symbol table. This function scans
// each object file's symbol table to copy local symbols to the output.
template <class ELFT> void Writer<ELFT>::copyLocalSymbols() {
  if (!In<ELFT>::SymTab)
    return;
  for (elf::ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles()) {
    for (SymbolBody *B : F->getLocalSymbols()) {
      if (!B->IsLocal)
        fatal(toString(F) +
              ": broken object: getLocalSymbols returns a non-local symbol");
      auto *DR = dyn_cast<DefinedRegular<ELFT>>(B);

      // No reason to keep local undefined symbol in symtab.
      if (!DR)
        continue;
      if (!includeInSymtab<ELFT>(*B))
        continue;

      InputSectionBase<ELFT> *Sec = DR->Section;
      if (!shouldKeepInSymtab<ELFT>(Sec, B->getName(), *B))
        continue;
      ++In<ELFT>::SymTab->NumLocals;
      if (Config->Relocatable)
        B->DynsymIndex = In<ELFT>::SymTab->NumLocals;
      F->KeptLocalSyms.push_back(std::make_pair(
          DR, In<ELFT>::SymTab->StrTabSec.addString(B->getName())));
    }
  }
}

// PPC64 has a number of special SHT_PROGBITS+SHF_ALLOC+SHF_WRITE sections that
// we would like to make sure appear is a specific order to maximize their
// coverage by a single signed 16-bit offset from the TOC base pointer.
// Conversely, the special .tocbss section should be first among all SHT_NOBITS
// sections. This will put it next to the loaded special PPC64 sections (and,
// thus, within reach of the TOC base pointer).
static int getPPC64SectionRank(StringRef SectionName) {
  return StringSwitch<int>(SectionName)
      .Case(".tocbss", 0)
      .Case(".branch_lt", 2)
      .Case(".toc", 3)
      .Case(".toc1", 4)
      .Case(".opd", 5)
      .Default(1);
}

template <class ELFT> bool elf::isRelroSection(const OutputSectionBase *Sec) {
  if (!Config->ZRelro)
    return false;
  uint64_t Flags = Sec->Flags;
  if (!(Flags & SHF_ALLOC) || !(Flags & SHF_WRITE))
    return false;
  if (Flags & SHF_TLS)
    return true;
  uint32_t Type = Sec->Type;
  if (Type == SHT_INIT_ARRAY || Type == SHT_FINI_ARRAY ||
      Type == SHT_PREINIT_ARRAY)
    return true;
  if (Sec == In<ELFT>::GotPlt->OutSec)
    return Config->ZNow;
  if (Sec == In<ELFT>::Dynamic->OutSec)
    return true;
  if (In<ELFT>::Got && Sec == In<ELFT>::Got->OutSec)
    return true;
  if (In<ELFT>::MipsGot && Sec == In<ELFT>::MipsGot->OutSec)
    return true;
  StringRef S = Sec->getName();
  return S == ".data.rel.ro" || S == ".ctors" || S == ".dtors" || S == ".jcr" ||
         S == ".eh_frame";
}

template <class ELFT>
static bool compareSectionsNonScript(const OutputSectionBase *A,
                                     const OutputSectionBase *B) {
  // Put .interp first because some loaders want to see that section
  // on the first page of the executable file when loaded into memory.
  bool AIsInterp = A->getName() == ".interp";
  bool BIsInterp = B->getName() == ".interp";
  if (AIsInterp != BIsInterp)
    return AIsInterp;

  // Allocatable sections go first to reduce the total PT_LOAD size and
  // so debug info doesn't change addresses in actual code.
  bool AIsAlloc = A->Flags & SHF_ALLOC;
  bool BIsAlloc = B->Flags & SHF_ALLOC;
  if (AIsAlloc != BIsAlloc)
    return AIsAlloc;

  // We don't have any special requirements for the relative order of two non
  // allocatable sections.
  if (!AIsAlloc)
    return false;

  // We want the read only sections first so that they go in the PT_LOAD
  // covering the program headers at the start of the file.
  bool AIsWritable = A->Flags & SHF_WRITE;
  bool BIsWritable = B->Flags & SHF_WRITE;
  if (AIsWritable != BIsWritable)
    return BIsWritable;

  if (!Config->SingleRoRx) {
    // For a corresponding reason, put non exec sections first (the program
    // header PT_LOAD is not executable).
    // We only do that if we are not using linker scripts, since with linker
    // scripts ro and rx sections are in the same PT_LOAD, so their relative
    // order is not important. The same applies for -no-rosegment.
    bool AIsExec = A->Flags & SHF_EXECINSTR;
    bool BIsExec = B->Flags & SHF_EXECINSTR;
    if (AIsExec != BIsExec)
      return BIsExec;
  }

  // If we got here we know that both A and B are in the same PT_LOAD.

  // The TLS initialization block needs to be a single contiguous block in a R/W
  // PT_LOAD, so stick TLS sections directly before R/W sections. The TLS NOBITS
  // sections are placed here as they don't take up virtual address space in the
  // PT_LOAD.
  bool AIsTls = A->Flags & SHF_TLS;
  bool BIsTls = B->Flags & SHF_TLS;
  if (AIsTls != BIsTls)
    return AIsTls;

  // The next requirement we have is to put nobits sections last. The
  // reason is that the only thing the dynamic linker will see about
  // them is a p_memsz that is larger than p_filesz. Seeing that it
  // zeros the end of the PT_LOAD, so that has to correspond to the
  // nobits sections.
  bool AIsNoBits = A->Type == SHT_NOBITS;
  bool BIsNoBits = B->Type == SHT_NOBITS;
  if (AIsNoBits != BIsNoBits)
    return BIsNoBits;

  // We place RelRo section before plain r/w ones.
  bool AIsRelRo = isRelroSection<ELFT>(A);
  bool BIsRelRo = isRelroSection<ELFT>(B);
  if (AIsRelRo != BIsRelRo)
    return AIsRelRo;

  // Some architectures have additional ordering restrictions for sections
  // within the same PT_LOAD.
  if (Config->EMachine == EM_PPC64)
    return getPPC64SectionRank(A->getName()) <
           getPPC64SectionRank(B->getName());

  return false;
}

// Output section ordering is determined by this function.
template <class ELFT>
static bool compareSections(const OutputSectionBase *A,
                            const OutputSectionBase *B) {
  // For now, put sections mentioned in a linker script first.
  int AIndex = Script<ELFT>::X->getSectionIndex(A->getName());
  int BIndex = Script<ELFT>::X->getSectionIndex(B->getName());
  bool AInScript = AIndex != INT_MAX;
  bool BInScript = BIndex != INT_MAX;
  if (AInScript != BInScript)
    return AInScript;
  // If both are in the script, use that order.
  if (AInScript)
    return AIndex < BIndex;

  return compareSectionsNonScript<ELFT>(A, B);
}

// Program header entry
template <class ELFT>
PhdrEntry<ELFT>::PhdrEntry(unsigned Type, unsigned Flags) {
  H.p_type = Type;
  H.p_flags = Flags;
}

template <class ELFT> void PhdrEntry<ELFT>::add(OutputSectionBase *Sec) {
  Last = Sec;
  if (!First)
    First = Sec;
  H.p_align = std::max<typename ELFT::uint>(H.p_align, Sec->Addralign);
  if (H.p_type == PT_LOAD)
    Sec->FirstInPtLoad = First;
}

template <class ELFT>
static Symbol *addOptionalSynthetic(StringRef Name, OutputSectionBase *Sec,
                                    typename ELFT::uint Val,
                                    uint8_t StOther = STV_HIDDEN) {
  SymbolBody *S = Symtab<ELFT>::X->find(Name);
  if (!S)
    return nullptr;
  if (!S->isUndefined() && !S->isShared())
    return S->symbol();
  return Symtab<ELFT>::X->addSynthetic(Name, Sec, Val, StOther);
}

template <class ELFT>
static Symbol *addRegular(StringRef Name, InputSectionBase<ELFT> *Sec,
                          typename ELFT::uint Value) {
  // The linker generated symbols are added as STB_WEAK to allow user defined
  // ones to override them.
  return Symtab<ELFT>::X->addRegular(Name, STV_HIDDEN, STT_NOTYPE, Value,
                                     /*Size=*/0, STB_WEAK, Sec,
                                     /*File=*/nullptr);
}

template <class ELFT>
static Symbol *addOptionalRegular(StringRef Name, InputSectionBase<ELFT> *IS,
                                  typename ELFT::uint Value) {
  SymbolBody *S = Symtab<ELFT>::X->find(Name);
  if (!S)
    return nullptr;
  if (!S->isUndefined() && !S->isShared())
    return S->symbol();
  return addRegular(Name, IS, Value);
}

// The beginning and the ending of .rel[a].plt section are marked
// with __rel[a]_iplt_{start,end} symbols if it is a statically linked
// executable. The runtime needs these symbols in order to resolve
// all IRELATIVE relocs on startup. For dynamic executables, we don't
// need these symbols, since IRELATIVE relocs are resolved through GOT
// and PLT. For details, see http://www.airs.com/blog/archives/403.
template <class ELFT> void Writer<ELFT>::addRelIpltSymbols() {
  if (In<ELFT>::DynSymTab)
    return;
  StringRef S = Config->Rela ? "__rela_iplt_start" : "__rel_iplt_start";
  addOptionalRegular<ELFT>(S, In<ELFT>::RelaPlt, 0);

  S = Config->Rela ? "__rela_iplt_end" : "__rel_iplt_end";
  addOptionalRegular<ELFT>(S, In<ELFT>::RelaPlt, -1);
}

// The linker is expected to define some symbols depending on
// the linking result. This function defines such symbols.
template <class ELFT> void Writer<ELFT>::addReservedSymbols() {
  if (Config->EMachine == EM_MIPS) {
    // Define _gp for MIPS. st_value of _gp symbol will be updated by Writer
    // so that it points to an absolute address which is relative to GOT.
    // Default offset is 0x7ff0.
    // See "Global Data Symbols" in Chapter 6 in the following document:
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    ElfSym<ELFT>::MipsGp = addRegular("_gp", In<ELFT>::MipsGot, 0x7ff0)->body();

    // On MIPS O32 ABI, _gp_disp is a magic symbol designates offset between
    // start of function and 'gp' pointer into GOT.
    if (Symbol *S = addOptionalRegular("_gp_disp", In<ELFT>::MipsGot, 0))
      ElfSym<ELFT>::MipsGpDisp = cast<DefinedRegular<ELFT>>(S->body());

    // The __gnu_local_gp is a magic symbol equal to the current value of 'gp'
    // pointer. This symbol is used in the code generated by .cpload pseudo-op
    // in case of using -mno-shared option.
    // https://sourceware.org/ml/binutils/2004-12/msg00094.html
    if (Symbol *S = addOptionalRegular("__gnu_local_gp", In<ELFT>::MipsGot, 0))
      ElfSym<ELFT>::MipsLocalGp = cast<DefinedRegular<ELFT>>(S->body());
  }

  // In the assembly for 32 bit x86 the _GLOBAL_OFFSET_TABLE_ symbol
  // is magical and is used to produce a R_386_GOTPC relocation.
  // The R_386_GOTPC relocation value doesn't actually depend on the
  // symbol value, so it could use an index of STN_UNDEF which, according
  // to the spec, means the symbol value is 0.
  // Unfortunately both gas and MC keep the _GLOBAL_OFFSET_TABLE_ symbol in
  // the object file.
  // The situation is even stranger on x86_64 where the assembly doesn't
  // need the magical symbol, but gas still puts _GLOBAL_OFFSET_TABLE_ as
  // an undefined symbol in the .o files.
  // Given that the symbol is effectively unused, we just create a dummy
  // hidden one to avoid the undefined symbol error.
  Symtab<ELFT>::X->addIgnored("_GLOBAL_OFFSET_TABLE_");

  // __tls_get_addr is defined by the dynamic linker for dynamic ELFs. For
  // static linking the linker is required to optimize away any references to
  // __tls_get_addr, so it's not defined anywhere. Create a hidden definition
  // to avoid the undefined symbol error. As usual special cases are ARM and
  // MIPS - the libc for these targets defines __tls_get_addr itself because
  // there are no TLS optimizations for these targets.
  if (!In<ELFT>::DynSymTab &&
      (Config->EMachine != EM_MIPS && Config->EMachine != EM_ARM))
    Symtab<ELFT>::X->addIgnored("__tls_get_addr");

  // If linker script do layout we do not need to create any standart symbols.
  if (ScriptConfig->HasSections)
    return;

  ElfSym<ELFT>::EhdrStart = Symtab<ELFT>::X->addIgnored("__ehdr_start");

  auto Define = [this](StringRef S, DefinedRegular<ELFT> *&Sym1,
                       DefinedRegular<ELFT> *&Sym2) {
    Sym1 = Symtab<ELFT>::X->addIgnored(S, STV_DEFAULT);

    // The name without the underscore is not a reserved name,
    // so it is defined only when there is a reference against it.
    assert(S.startswith("_"));
    S = S.substr(1);
    if (SymbolBody *B = Symtab<ELFT>::X->find(S))
      if (B->isUndefined())
        Sym2 = Symtab<ELFT>::X->addAbsolute(S, STV_DEFAULT);
  };

  Define("_end", ElfSym<ELFT>::End, ElfSym<ELFT>::End2);
  Define("_etext", ElfSym<ELFT>::Etext, ElfSym<ELFT>::Etext2);
  Define("_edata", ElfSym<ELFT>::Edata, ElfSym<ELFT>::Edata2);
}

// Sort input sections by section name suffixes for
// __attribute__((init_priority(N))).
template <class ELFT> static void sortInitFini(OutputSectionBase *S) {
  if (S)
    reinterpret_cast<OutputSection<ELFT> *>(S)->sortInitFini();
}

// Sort input sections by the special rule for .ctors and .dtors.
template <class ELFT> static void sortCtorsDtors(OutputSectionBase *S) {
  if (S)
    reinterpret_cast<OutputSection<ELFT> *>(S)->sortCtorsDtors();
}

// Sort input sections using the list provided by --symbol-ordering-file.
template <class ELFT>
static void sortBySymbolsOrder(ArrayRef<OutputSectionBase *> V) {
  if (Config->SymbolOrderingFile.empty())
    return;

  // Build sections order map from symbols list.
  DenseMap<InputSectionBase<ELFT> *, unsigned> SectionsOrder;
  for (elf::ObjectFile<ELFT> *File : Symtab<ELFT>::X->getObjectFiles()) {
    for (SymbolBody *Body : File->getSymbols()) {
      auto *D = dyn_cast<DefinedRegular<ELFT>>(Body);
      if (!D || !D->Section)
        continue;
      auto It =
          Config->SymbolOrderingFile.find(CachedHashString(Body->getName()));
      if (It == Config->SymbolOrderingFile.end())
        continue;

      auto It2 = SectionsOrder.insert({D->Section, It->second});
      if (!It2.second)
        It2.first->second = std::min(It->second, It2.first->second);
    }
  }

  for (OutputSectionBase *Base : V)
    if (OutputSection<ELFT> *Sec = dyn_cast<OutputSection<ELFT>>(Base))
      Sec->sort([&](InputSection<ELFT> *S) {
        auto It = SectionsOrder.find(S);
        return It == SectionsOrder.end() ? UINT32_MAX : It->second;
      });
}

template <class ELFT>
void Writer<ELFT>::forEachRelSec(
    std::function<void(InputSectionBase<ELFT> &)> Fn) {
  for (InputSectionBase<ELFT> *IS : Symtab<ELFT>::X->Sections) {
    if (!IS->Live)
      continue;
    // Scan all relocations. Each relocation goes through a series
    // of tests to determine if it needs special treatment, such as
    // creating GOT, PLT, copy relocations, etc.
    // Note that relocations for non-alloc sections are directly
    // processed by InputSection::relocateNonAlloc.
    if (!(IS->Flags & SHF_ALLOC))
      continue;
    if (isa<InputSection<ELFT>>(IS) || isa<EhInputSection<ELFT>>(IS))
      Fn(*IS);
  }
}

template <class ELFT>
void Writer<ELFT>::addInputSec(InputSectionBase<ELFT> *IS) {
  if (!IS)
    return;

  if (!IS->Live) {
    reportDiscarded(IS);
    return;
  }
  OutputSectionBase *Sec;
  bool IsNew;
  StringRef OutsecName = getOutputSectionName(IS->Name);
  std::tie(Sec, IsNew) = Factory.create(IS, OutsecName);
  if (IsNew)
    OutputSections.push_back(Sec);
  Sec->addSection(IS);
}

template <class ELFT> void Writer<ELFT>::createSections() {
  for (InputSectionBase<ELFT> *IS : Symtab<ELFT>::X->Sections)
    addInputSec(IS);

  sortBySymbolsOrder<ELFT>(OutputSections);
  sortInitFini<ELFT>(findSection(".init_array"));
  sortInitFini<ELFT>(findSection(".fini_array"));
  sortCtorsDtors<ELFT>(findSection(".ctors"));
  sortCtorsDtors<ELFT>(findSection(".dtors"));

  for (OutputSectionBase *Sec : OutputSections)
    Sec->assignOffsets();
}

template <class ELFT>
static bool canSharePtLoad(const OutputSectionBase &S1,
                           const OutputSectionBase &S2) {
  if (!(S1.Flags & SHF_ALLOC) || !(S2.Flags & SHF_ALLOC))
    return false;

  bool S1IsWrite = S1.Flags & SHF_WRITE;
  bool S2IsWrite = S2.Flags & SHF_WRITE;
  if (S1IsWrite != S2IsWrite)
    return false;

  if (!S1IsWrite)
    return true; // RO and RX share a PT_LOAD with linker scripts.
  return (S1.Flags & SHF_EXECINSTR) == (S2.Flags & SHF_EXECINSTR);
}

template <class ELFT> void Writer<ELFT>::sortSections() {
  // Don't sort if using -r. It is not necessary and we want to preserve the
  // relative order for SHF_LINK_ORDER sections.
  if (Config->Relocatable)
    return;
  if (!ScriptConfig->HasSections) {
    std::stable_sort(OutputSections.begin(), OutputSections.end(),
                     compareSectionsNonScript<ELFT>);
    return;
  }
  Script<ELFT>::X->adjustSectionsBeforeSorting();

  // The order of the sections in the script is arbitrary and may not agree with
  // compareSectionsNonScript. This means that we cannot easily define a
  // strict weak ordering. To see why, consider a comparison of a section in the
  // script and one not in the script. We have a two simple options:
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
  // *  First put script sections at the start and sort the script and
  //    non-script sections independently.
  // *  Move each non-script section to its preferred position. We try
  //    to put each section in the last position where it it can share
  //    a PT_LOAD.

  std::stable_sort(OutputSections.begin(), OutputSections.end(),
                   compareSections<ELFT>);

  auto I = OutputSections.begin();
  auto E = OutputSections.end();
  auto NonScriptI =
      std::find_if(OutputSections.begin(), E, [](OutputSectionBase *S) {
        return Script<ELFT>::X->getSectionIndex(S->getName()) == INT_MAX;
      });
  while (NonScriptI != E) {
    auto BestPos = std::max_element(
        I, NonScriptI, [&](OutputSectionBase *&A, OutputSectionBase *&B) {
          bool ACanSharePtLoad = canSharePtLoad<ELFT>(**NonScriptI, *A);
          bool BCanSharePtLoad = canSharePtLoad<ELFT>(**NonScriptI, *B);
          if (ACanSharePtLoad != BCanSharePtLoad)
            return BCanSharePtLoad;

          bool ACmp = compareSectionsNonScript<ELFT>(*NonScriptI, A);
          bool BCmp = compareSectionsNonScript<ELFT>(*NonScriptI, B);
          if (ACmp != BCmp)
            return BCmp; // FIXME: missing test

          size_t PosA = &A - &OutputSections[0];
          size_t PosB = &B - &OutputSections[0];
          return ACmp ? PosA > PosB : PosA < PosB;
        });

    // max_element only returns NonScriptI if the range is empty. If the range
    // is not empty we should consider moving the the element forward one
    // position.
    if (BestPos != NonScriptI &&
        !compareSectionsNonScript<ELFT>(*NonScriptI, *BestPos))
      ++BestPos;
    std::rotate(BestPos, NonScriptI, NonScriptI + 1);
    ++NonScriptI;
  }

  Script<ELFT>::X->adjustSectionsAfterSorting();
}

template <class ELFT>
static void
finalizeSynthetic(const std::vector<SyntheticSection<ELFT> *> &Sections) {
  for (SyntheticSection<ELFT> *SS : Sections)
    if (SS && SS->OutSec && !SS->empty()) {
      SS->finalize();
      SS->OutSec->Size = 0;
      SS->OutSec->assignOffsets();
    }
}

// We need to add input synthetic sections early in createSyntheticSections()
// to make them visible from linkescript side. But not all sections are always
// required to be in output. For example we don't need dynamic section content
// sometimes. This function filters out such unused sections from output.
template <class ELFT>
static void removeUnusedSyntheticSections(std::vector<OutputSectionBase *> &V) {
  // Input synthetic sections are placed after all regular ones. We iterate over
  // them all and exit at first non-synthetic.
  for (InputSectionBase<ELFT> *S : llvm::reverse(Symtab<ELFT>::X->Sections)) {
    SyntheticSection<ELFT> *SS = dyn_cast<SyntheticSection<ELFT>>(S);
    if (!SS)
      return;
    if (!SS->empty() || !SS->OutSec)
      continue;

    OutputSection<ELFT> *OutSec = cast<OutputSection<ELFT>>(SS->OutSec);
    OutSec->Sections.erase(
        std::find(OutSec->Sections.begin(), OutSec->Sections.end(), SS));
    // If there is no other sections in output section, remove it from output.
    if (OutSec->Sections.empty())
      V.erase(std::find(V.begin(), V.end(), OutSec));
  }
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::finalizeSections() {
  Out<ELFT>::DebugInfo = findSection(".debug_info");
  Out<ELFT>::PreinitArray = findSection(".preinit_array");
  Out<ELFT>::InitArray = findSection(".init_array");
  Out<ELFT>::FiniArray = findSection(".fini_array");

  // The linker needs to define SECNAME_start, SECNAME_end and SECNAME_stop
  // symbols for sections, so that the runtime can get the start and end
  // addresses of each section by section name. Add such symbols.
  if (!Config->Relocatable) {
    addStartEndSymbols();
    for (OutputSectionBase *Sec : OutputSections)
      addStartStopSymbols(Sec);
  }

  // Add _DYNAMIC symbol. Unlike GNU gold, our _DYNAMIC symbol has no type.
  // It should be okay as no one seems to care about the type.
  // Even the author of gold doesn't remember why gold behaves that way.
  // https://sourceware.org/ml/binutils/2002-03/msg00360.html
  if (In<ELFT>::DynSymTab)
    addRegular("_DYNAMIC", In<ELFT>::Dynamic, 0);

  // Define __rel[a]_iplt_{start,end} symbols if needed.
  addRelIpltSymbols();

  if (!Out<ELFT>::EhFrame->empty()) {
    OutputSections.push_back(Out<ELFT>::EhFrame);
    Out<ELFT>::EhFrame->finalize();
  }

  // Scan relocations. This must be done after every symbol is declared so that
  // we can correctly decide if a dynamic relocation is needed.
  forEachRelSec(scanRelocations<ELFT>);

  // Now that we have defined all possible symbols including linker-
  // synthesized ones. Visit all symbols to give the finishing touches.
  for (Symbol *S : Symtab<ELFT>::X->getSymbols()) {
    SymbolBody *Body = S->body();

    if (!includeInSymtab<ELFT>(*Body))
      continue;
    if (In<ELFT>::SymTab)
      In<ELFT>::SymTab->addSymbol(Body);

    if (In<ELFT>::DynSymTab && S->includeInDynsym()) {
      In<ELFT>::DynSymTab->addSymbol(Body);
      if (auto *SS = dyn_cast<SharedSymbol<ELFT>>(Body))
        if (SS->file()->isNeeded())
          In<ELFT>::VerNeed->addSymbol(SS);
    }
  }

  // Do not proceed if there was an undefined symbol.
  if (ErrorCount)
    return;

  // So far we have added sections from input object files.
  // This function adds linker-created Out<ELFT>::* sections.
  addPredefinedSections();
  removeUnusedSyntheticSections<ELFT>(OutputSections);

  sortSections();

  unsigned I = 1;
  for (OutputSectionBase *Sec : OutputSections) {
    Sec->SectionIndex = I++;
    Sec->ShName = In<ELFT>::ShStrTab->addString(Sec->getName());
  }

  // Fill other section headers. The dynamic table is finalized
  // at the end because some tags like RELSZ depend on result
  // of finalizing other sections.
  for (OutputSectionBase *Sec : OutputSections)
    Sec->finalize();

  // Dynamic section must be the last one in this list and dynamic
  // symbol table section (DynSymTab) must be the first one.
  finalizeSynthetic<ELFT>(
      {In<ELFT>::DynSymTab, In<ELFT>::GnuHashTab, In<ELFT>::HashTab,
       In<ELFT>::SymTab, In<ELFT>::ShStrTab, In<ELFT>::StrTab, In<ELFT>::VerDef,
       In<ELFT>::DynStrTab, In<ELFT>::GdbIndex, In<ELFT>::Got,
       In<ELFT>::MipsGot, In<ELFT>::GotPlt, In<ELFT>::RelaDyn,
       In<ELFT>::RelaPlt, In<ELFT>::Plt, In<ELFT>::EhFrameHdr, In<ELFT>::VerSym,
       In<ELFT>::VerNeed, In<ELFT>::Dynamic});
}

template <class ELFT> void Writer<ELFT>::addPredefinedSections() {
  if (Out<ELFT>::Bss->Size > 0)
    OutputSections.push_back(Out<ELFT>::Bss);

  auto OS = dyn_cast_or_null<OutputSection<ELFT>>(findSection(".ARM.exidx"));
  if (OS && !OS->Sections.empty() && !Config->Relocatable)
    OS->addSection(make<ARMExidxSentinelSection<ELFT>>());

  addInputSec(In<ELFT>::SymTab);
  addInputSec(In<ELFT>::ShStrTab);
  addInputSec(In<ELFT>::StrTab);
}

// The linker is expected to define SECNAME_start and SECNAME_end
// symbols for a few sections. This function defines them.
template <class ELFT> void Writer<ELFT>::addStartEndSymbols() {
  auto Define = [&](StringRef Start, StringRef End, OutputSectionBase *OS) {
    // These symbols resolve to the image base if the section does not exist.
    addOptionalSynthetic<ELFT>(Start, OS, 0);
    addOptionalSynthetic<ELFT>(End, OS,
                               OS ? DefinedSynthetic<ELFT>::SectionEnd : 0);
  };

  Define("__preinit_array_start", "__preinit_array_end",
         Out<ELFT>::PreinitArray);
  Define("__init_array_start", "__init_array_end", Out<ELFT>::InitArray);
  Define("__fini_array_start", "__fini_array_end", Out<ELFT>::FiniArray);

  if (OutputSectionBase *Sec = findSection(".ARM.exidx"))
    Define("__exidx_start", "__exidx_end", Sec);
}

// If a section name is valid as a C identifier (which is rare because of
// the leading '.'), linkers are expected to define __start_<secname> and
// __stop_<secname> symbols. They are at beginning and end of the section,
// respectively. This is not requested by the ELF standard, but GNU ld and
// gold provide the feature, and used by many programs.
template <class ELFT>
void Writer<ELFT>::addStartStopSymbols(OutputSectionBase *Sec) {
  StringRef S = Sec->getName();
  if (!isValidCIdentifier(S))
    return;
  addOptionalSynthetic<ELFT>(Saver.save("__start_" + S), Sec, 0, STV_DEFAULT);
  addOptionalSynthetic<ELFT>(Saver.save("__stop_" + S), Sec,
                             DefinedSynthetic<ELFT>::SectionEnd, STV_DEFAULT);
}

template <class ELFT>
OutputSectionBase *Writer<ELFT>::findSection(StringRef Name) {
  for (OutputSectionBase *Sec : OutputSections)
    if (Sec->getName() == Name)
      return Sec;
  return nullptr;
}

template <class ELFT> static bool needsPtLoad(OutputSectionBase *Sec) {
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
template <class ELFT>
static typename ELFT::uint computeFlags(typename ELFT::uint F) {
  if (Config->OMagic)
    return PF_R | PF_W | PF_X;
  if (Config->SingleRoRx && !(F & PF_W))
    return F | PF_X;
  return F;
}

// Decide which program headers to create and which sections to include in each
// one.
template <class ELFT> std::vector<PhdrEntry<ELFT>> Writer<ELFT>::createPhdrs() {
  std::vector<Phdr> Ret;
  auto AddHdr = [&](unsigned Type, unsigned Flags) -> Phdr * {
    Ret.emplace_back(Type, Flags);
    return &Ret.back();
  };

  // The first phdr entry is PT_PHDR which describes the program header itself.
  Phdr &Hdr = *AddHdr(PT_PHDR, PF_R);
  Hdr.add(Out<ELFT>::ProgramHeaders);

  // PT_INTERP must be the second entry if exists.
  if (OutputSectionBase *Sec = findSection(".interp")) {
    Phdr &Hdr = *AddHdr(PT_INTERP, Sec->getPhdrFlags());
    Hdr.add(Sec);
  }

  // Add the first PT_LOAD segment for regular output sections.
  uintX_t Flags = computeFlags<ELFT>(PF_R);
  Phdr *Load = AddHdr(PT_LOAD, Flags);
  if (!ScriptConfig->HasSections) {
    Load->add(Out<ELFT>::ElfHeader);
    Load->add(Out<ELFT>::ProgramHeaders);
  }

  Phdr TlsHdr(PT_TLS, PF_R);
  Phdr RelRo(PT_GNU_RELRO, PF_R);
  Phdr Note(PT_NOTE, PF_R);
  for (OutputSectionBase *Sec : OutputSections) {
    if (!(Sec->Flags & SHF_ALLOC))
      break;

    // If we meet TLS section then we create TLS header
    // and put all TLS sections inside for further use when
    // assign addresses.
    if (Sec->Flags & SHF_TLS)
      TlsHdr.add(Sec);

    if (!needsPtLoad<ELFT>(Sec))
      continue;

    // Segments are contiguous memory regions that has the same attributes
    // (e.g. executable or writable). There is one phdr for each segment.
    // Therefore, we need to create a new phdr when the next section has
    // different flags or is loaded at a discontiguous address using AT linker
    // script command.
    uintX_t NewFlags = computeFlags<ELFT>(Sec->getPhdrFlags());
    if (Script<ELFT>::X->hasLMA(Sec->getName()) || Flags != NewFlags) {
      Load = AddHdr(PT_LOAD, NewFlags);
      Flags = NewFlags;
    }

    Load->add(Sec);

    if (isRelroSection<ELFT>(Sec))
      RelRo.add(Sec);
    if (Sec->Type == SHT_NOTE)
      Note.add(Sec);
  }

  // Add the TLS segment unless it's empty.
  if (TlsHdr.First)
    Ret.push_back(std::move(TlsHdr));

  // Add an entry for .dynamic.
  if (In<ELFT>::DynSymTab) {
    Phdr &H = *AddHdr(PT_DYNAMIC, In<ELFT>::Dynamic->OutSec->getPhdrFlags());
    H.add(In<ELFT>::Dynamic->OutSec);
  }

  // PT_GNU_RELRO includes all sections that should be marked as
  // read-only by dynamic linker after proccessing relocations.
  if (RelRo.First)
    Ret.push_back(std::move(RelRo));

  // PT_GNU_EH_FRAME is a special section pointing on .eh_frame_hdr.
  if (!Out<ELFT>::EhFrame->empty() && In<ELFT>::EhFrameHdr) {
    Phdr &Hdr =
        *AddHdr(PT_GNU_EH_FRAME, In<ELFT>::EhFrameHdr->OutSec->getPhdrFlags());
    Hdr.add(In<ELFT>::EhFrameHdr->OutSec);
  }

  // PT_OPENBSD_RANDOMIZE specifies the location and size of a part of the
  // memory image of the program that must be filled with random data before any
  // code in the object is executed.
  if (OutputSectionBase *Sec = findSection(".openbsd.randomdata")) {
    Phdr &Hdr = *AddHdr(PT_OPENBSD_RANDOMIZE, Sec->getPhdrFlags());
    Hdr.add(Sec);
  }

  // PT_GNU_STACK is a special section to tell the loader to make the
  // pages for the stack non-executable.
  if (!Config->ZExecstack) {
    Phdr &Hdr = *AddHdr(PT_GNU_STACK, PF_R | PF_W);
    if (Config->ZStackSize != uint64_t(-1))
      Hdr.H.p_memsz = Config->ZStackSize;
  }

  // PT_OPENBSD_WXNEEDED is a OpenBSD-specific header to mark the executable
  // is expected to perform W^X violations, such as calling mprotect(2) or
  // mmap(2) with PROT_WRITE | PROT_EXEC, which is prohibited by default on
  // OpenBSD.
  if (Config->ZWxneeded)
    AddHdr(PT_OPENBSD_WXNEEDED, PF_X);

  if (Note.First)
    Ret.push_back(std::move(Note));
  return Ret;
}

template <class ELFT>
void Writer<ELFT>::addPtArmExid(std::vector<PhdrEntry<ELFT>> &Phdrs) {
  if (Config->EMachine != EM_ARM)
    return;
  auto I = std::find_if(
      OutputSections.begin(), OutputSections.end(),
      [](OutputSectionBase *Sec) { return Sec->Type == SHT_ARM_EXIDX; });
  if (I == OutputSections.end())
    return;

  // PT_ARM_EXIDX is the ARM EHABI equivalent of PT_GNU_EH_FRAME
  Phdr ARMExidx(PT_ARM_EXIDX, PF_R);
  ARMExidx.add(*I);
  Phdrs.push_back(ARMExidx);
}

// The first section of each PT_LOAD and the first section after PT_GNU_RELRO
// have to be page aligned so that the dynamic linker can set the permissions.
template <class ELFT> void Writer<ELFT>::fixSectionAlignments() {
  for (const Phdr &P : Phdrs)
    if (P.H.p_type == PT_LOAD)
      P.First->PageAlign = true;

  for (const Phdr &P : Phdrs) {
    if (P.H.p_type != PT_GNU_RELRO)
      continue;
    // Find the first section after PT_GNU_RELRO. If it is in a PT_LOAD we
    // have to align it to a page.
    auto End = OutputSections.end();
    auto I = std::find(OutputSections.begin(), End, P.Last);
    if (I == End || (I + 1) == End)
      continue;
    OutputSectionBase *Sec = *(I + 1);
    if (needsPtLoad<ELFT>(Sec))
      Sec->PageAlign = true;
  }
}

// We should set file offsets and VAs for elf header and program headers
// sections. These are special, we do not include them into output sections
// list, but have them to simplify the code.
template <class ELFT> void Writer<ELFT>::fixHeaders() {
  Out<ELFT>::ProgramHeaders->Size = sizeof(Elf_Phdr) * Phdrs.size();
  // If the script has SECTIONS, assignAddresses will compute the values.
  if (ScriptConfig->HasSections)
    return;
  uintX_t BaseVA = Config->ImageBase;
  Out<ELFT>::ElfHeader->Addr = BaseVA;
  Out<ELFT>::ProgramHeaders->Addr = BaseVA + Out<ELFT>::ElfHeader->Size;
}

// Assign VAs (addresses at run-time) to output sections.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  uintX_t VA = Config->ImageBase + getHeaderSize<ELFT>();
  uintX_t ThreadBssOffset = 0;
  for (OutputSectionBase *Sec : OutputSections) {
    uintX_t Alignment = Sec->Addralign;
    if (Sec->PageAlign)
      Alignment = std::max<uintX_t>(Alignment, Config->MaxPageSize);

    auto I = Config->SectionStartMap.find(Sec->getName());
    if (I != Config->SectionStartMap.end())
      VA = I->second;

    // We only assign VAs to allocated sections.
    if (needsPtLoad<ELFT>(Sec)) {
      VA = alignTo(VA, Alignment);
      Sec->Addr = VA;
      VA += Sec->Size;
    } else if (Sec->Flags & SHF_TLS && Sec->Type == SHT_NOBITS) {
      uintX_t TVA = VA + ThreadBssOffset;
      TVA = alignTo(TVA, Alignment);
      Sec->Addr = TVA;
      ThreadBssOffset = TVA - VA + Sec->Size;
    }
  }
}

// Adjusts the file alignment for a given output section and returns
// its new file offset. The file offset must be the same with its
// virtual address (modulo the page size) so that the loader can load
// executables without any address adjustment.
template <class ELFT, class uintX_t>
static uintX_t getFileAlignment(uintX_t Off, OutputSectionBase *Sec) {
  OutputSectionBase *First = Sec->FirstInPtLoad;
  // If the section is not in a PT_LOAD, we just have to align it.
  if (!First)
    return alignTo(Off, Sec->Addralign);

  // The first section in a PT_LOAD has to have congruent offset and address
  // module the page size.
  if (Sec == First)
    return alignTo(Off, Config->MaxPageSize, Sec->Addr);

  // If two sections share the same PT_LOAD the file offset is calculated
  // using this formula: Off2 = Off1 + (VA2 - VA1).
  return First->Offset + Sec->Addr - First->Addr;
}

template <class ELFT, class uintX_t>
void setOffset(OutputSectionBase *Sec, uintX_t &Off) {
  if (Sec->Type == SHT_NOBITS) {
    Sec->Offset = Off;
    return;
  }

  Off = getFileAlignment<ELFT>(Off, Sec);
  Sec->Offset = Off;
  Off += Sec->Size;
}

template <class ELFT> void Writer<ELFT>::assignFileOffsetsBinary() {
  uintX_t Off = 0;
  for (OutputSectionBase *Sec : OutputSections)
    if (Sec->Flags & SHF_ALLOC)
      setOffset<ELFT>(Sec, Off);
  FileSize = alignTo(Off, sizeof(uintX_t));
}

// Assign file offsets to output sections.
template <class ELFT> void Writer<ELFT>::assignFileOffsets() {
  uintX_t Off = 0;
  setOffset<ELFT>(Out<ELFT>::ElfHeader, Off);
  setOffset<ELFT>(Out<ELFT>::ProgramHeaders, Off);

  for (OutputSectionBase *Sec : OutputSections)
    setOffset<ELFT>(Sec, Off);

  SectionHeaderOff = alignTo(Off, sizeof(uintX_t));
  FileSize = SectionHeaderOff + (OutputSections.size() + 1) * sizeof(Elf_Shdr);
}

// Finalize the program headers. We call this function after we assign
// file offsets and VAs to all sections.
template <class ELFT> void Writer<ELFT>::setPhdrs() {
  for (Phdr &P : Phdrs) {
    Elf_Phdr &H = P.H;
    OutputSectionBase *First = P.First;
    OutputSectionBase *Last = P.Last;
    if (First) {
      H.p_filesz = Last->Offset - First->Offset;
      if (Last->Type != SHT_NOBITS)
        H.p_filesz += Last->Size;
      H.p_memsz = Last->Addr + Last->Size - First->Addr;
      H.p_offset = First->Offset;
      H.p_vaddr = First->Addr;
      if (!P.HasLMA)
        H.p_paddr = First->getLMA();
    }
    if (H.p_type == PT_LOAD)
      H.p_align = Config->MaxPageSize;
    else if (H.p_type == PT_GNU_RELRO)
      H.p_align = 1;

    // The TLS pointer goes after PT_TLS. At least glibc will align it,
    // so round up the size to make sure the offsets are correct.
    if (H.p_type == PT_TLS) {
      Out<ELFT>::TlsPhdr = &H;
      if (H.p_memsz)
        H.p_memsz = alignTo(H.p_memsz, H.p_align);
    }
  }
}

// The entry point address is chosen in the following ways.
//
// 1. the '-e' entry command-line option;
// 2. the ENTRY(symbol) command in a linker control script;
// 3. the value of the symbol start, if present;
// 4. the address of the first byte of the .text section, if present;
// 5. the address 0.
template <class ELFT> typename ELFT::uint Writer<ELFT>::getEntryAddr() {
  // Case 1, 2 or 3. As a special case, if the symbol is actually
  // a number, we'll use that number as an address.
  if (SymbolBody *B = Symtab<ELFT>::X->find(Config->Entry))
    return B->getVA<ELFT>();
  uint64_t Addr;
  if (!Config->Entry.getAsInteger(0, Addr))
    return Addr;

  // Case 4
  if (OutputSectionBase *Sec = findSection(".text")) {
    if (Config->WarnMissingEntry)
      warn("cannot find entry symbol " + Config->Entry + "; defaulting to 0x" +
           utohexstr(Sec->Addr));
    return Sec->Addr;
  }

  // Case 5
  if (Config->WarnMissingEntry)
    warn("cannot find entry symbol " + Config->Entry +
         "; not setting start address");
  return 0;
}

template <class ELFT> static uint8_t getELFEncoding() {
  if (ELFT::TargetEndianness == llvm::support::little)
    return ELFDATA2LSB;
  return ELFDATA2MSB;
}

static uint16_t getELFType() {
  if (Config->Pic)
    return ET_DYN;
  if (Config->Relocatable)
    return ET_REL;
  return ET_EXEC;
}

// This function is called after we have assigned address and size
// to each section. This function fixes some predefined absolute
// symbol values that depend on section address and size.
template <class ELFT> void Writer<ELFT>::fixAbsoluteSymbols() {
  // __ehdr_start is the location of program headers.
  if (ElfSym<ELFT>::EhdrStart)
    ElfSym<ELFT>::EhdrStart->Value = Out<ELFT>::ProgramHeaders->Addr;

  auto Set = [](DefinedRegular<ELFT> *S1, DefinedRegular<ELFT> *S2, uintX_t V) {
    if (S1)
      S1->Value = V;
    if (S2)
      S2->Value = V;
  };

  // _etext is the first location after the last read-only loadable segment.
  // _edata is the first location after the last read-write loadable segment.
  // _end is the first location after the uninitialized data region.
  for (Phdr &P : Phdrs) {
    Elf_Phdr &H = P.H;
    if (H.p_type != PT_LOAD)
      continue;
    Set(ElfSym<ELFT>::End, ElfSym<ELFT>::End2, H.p_vaddr + H.p_memsz);

    uintX_t Val = H.p_vaddr + H.p_filesz;
    if (H.p_flags & PF_W)
      Set(ElfSym<ELFT>::Edata, ElfSym<ELFT>::Edata2, Val);
    else
      Set(ElfSym<ELFT>::Etext, ElfSym<ELFT>::Etext2, Val);
  }

  // Setup MIPS _gp_disp/__gnu_local_gp symbols which should
  // be equal to the _gp symbol's value.
  if (Config->EMachine == EM_MIPS) {
    uintX_t GpDisp = In<ELFT>::MipsGot->getGp() - In<ELFT>::MipsGot->getVA();
    if (ElfSym<ELFT>::MipsGpDisp)
      ElfSym<ELFT>::MipsGpDisp->Value = GpDisp;
    if (ElfSym<ELFT>::MipsLocalGp)
      ElfSym<ELFT>::MipsLocalGp->Value = GpDisp;
  }
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  memcpy(Buf, "\177ELF", 4);

  // Write the ELF header.
  auto *EHdr = reinterpret_cast<Elf_Ehdr *>(Buf);
  EHdr->e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  EHdr->e_ident[EI_DATA] = getELFEncoding<ELFT>();
  EHdr->e_ident[EI_VERSION] = EV_CURRENT;
  EHdr->e_ident[EI_OSABI] = Config->OSABI;
  EHdr->e_type = getELFType();
  EHdr->e_machine = Config->EMachine;
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = getEntryAddr();
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_ehsize = sizeof(Elf_Ehdr);
  EHdr->e_phnum = Phdrs.size();
  EHdr->e_shentsize = sizeof(Elf_Shdr);
  EHdr->e_shnum = OutputSections.size() + 1;
  EHdr->e_shstrndx = In<ELFT>::ShStrTab->OutSec->SectionIndex;

  if (Config->EMachine == EM_ARM)
    // We don't currently use any features incompatible with EF_ARM_EABI_VER5,
    // but we don't have any firm guarantees of conformance. Linux AArch64
    // kernels (as of 2016) require an EABI version to be set.
    EHdr->e_flags = EF_ARM_EABI_VER5;
  else if (Config->EMachine == EM_MIPS)
    EHdr->e_flags = getMipsEFlags<ELFT>();

  if (!Config->Relocatable) {
    EHdr->e_phoff = sizeof(Elf_Ehdr);
    EHdr->e_phentsize = sizeof(Elf_Phdr);
  }

  // Write the program header table.
  auto *HBuf = reinterpret_cast<Elf_Phdr *>(Buf + EHdr->e_phoff);
  for (Phdr &P : Phdrs)
    *HBuf++ = P.H;

  // Write the section header table. Note that the first table entry is null.
  auto *SHdrs = reinterpret_cast<Elf_Shdr *>(Buf + EHdr->e_shoff);
  for (OutputSectionBase *Sec : OutputSections)
    Sec->writeHeaderTo<ELFT>(++SHdrs);
}

// Removes a given file asynchronously. This is a performance hack,
// so remove this when operating systems are improved.
//
// On Linux (and probably on other Unix-like systems), unlink(2) is a
// noticeably slow system call. As of 2016, unlink takes 250
// milliseconds to remove a 1 GB file on ext4 filesystem on my machine.
//
// To create a new result file, we first remove existing file. So, if
// you repeatedly link a 1 GB program in a regular compile-link-debug
// cycle, every cycle wastes 250 milliseconds only to remove a file.
// Since LLD can link a 1 GB binary in about 5 seconds, that waste
// actually counts.
//
// This function spawns a background thread to call unlink.
// The calling thread returns almost immediately.
static void unlinkAsync(StringRef Path) {
  if (!Config->Threads || !sys::fs::exists(Config->OutputFile))
    return;

  // First, rename Path to avoid race condition. We cannot remomve
  // Path from a different thread because we are now going to create
  // Path as a new file. If we do that in a different thread, the new
  // thread can remove the new file.
  SmallString<128> TempPath;
  if (auto EC = sys::fs::createUniqueFile(Path + "tmp%%%%%%%%", TempPath))
    fatal(EC, "createUniqueFile failed");
  if (auto EC = sys::fs::rename(Path, TempPath))
    fatal(EC, "rename failed");

  // Remove TempPath in background.
  std::thread([=] { ::remove(TempPath.str().str().c_str()); }).detach();
}

// Open a result file.
template <class ELFT> void Writer<ELFT>::openFile() {
  unlinkAsync(Config->OutputFile);
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Config->OutputFile, FileSize,
                               FileOutputBuffer::F_executable);

  if (auto EC = BufferOrErr.getError())
    error(EC, "failed to open " + Config->OutputFile);
  else
    Buffer = std::move(*BufferOrErr);
}

template <class ELFT> void Writer<ELFT>::writeSectionsBinary() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSectionBase *Sec : OutputSections)
    if (Sec->Flags & SHF_ALLOC)
      Sec->writeTo(Buf + Sec->Offset);
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();

  // PPC64 needs to process relocations in the .opd section
  // before processing relocations in code-containing sections.
  Out<ELFT>::Opd = findSection(".opd");
  if (Out<ELFT>::Opd) {
    Out<ELFT>::OpdBuf = Buf + Out<ELFT>::Opd->Offset;
    Out<ELFT>::Opd->writeTo(Buf + Out<ELFT>::Opd->Offset);
  }

  OutputSectionBase *EhFrameHdr =
      In<ELFT>::EhFrameHdr ? In<ELFT>::EhFrameHdr->OutSec : nullptr;
  for (OutputSectionBase *Sec : OutputSections)
    if (Sec != Out<ELFT>::Opd && Sec != EhFrameHdr)
      Sec->writeTo(Buf + Sec->Offset);

  // The .eh_frame_hdr depends on .eh_frame section contents, therefore
  // it should be written after .eh_frame is written.
  if (!Out<ELFT>::EhFrame->empty() && EhFrameHdr)
    EhFrameHdr->writeTo(Buf + EhFrameHdr->Offset);
}

template <class ELFT> void Writer<ELFT>::writeBuildId() {
  if (!In<ELFT>::BuildId || !In<ELFT>::BuildId->OutSec)
    return;

  // Compute a hash of all sections of the output file.
  uint8_t *Start = Buffer->getBufferStart();
  uint8_t *End = Start + FileSize;
  In<ELFT>::BuildId->writeBuildId({Start, End});
}

template <class ELFT> static std::string getErrorLoc(uint8_t *Loc) {
  for (InputSectionData *D : Symtab<ELFT>::X->Sections) {
    auto *IS = dyn_cast_or_null<InputSection<ELFT>>(D);
    if (!IS || !IS->OutSec)
      continue;

    uint8_t *ISLoc = cast<OutputSection<ELFT>>(IS->OutSec)->Loc + IS->OutSecOff;
    if (ISLoc <= Loc && ISLoc + IS->getSize() > Loc)
      return IS->getLocation(Loc - ISLoc) + ": ";
  }
  return "";
}

std::string elf::getErrorLocation(uint8_t *Loc) {
  switch (Config->EKind) {
  case ELF32LEKind:
    return getErrorLoc<ELF32LE>(Loc);
  case ELF32BEKind:
    return getErrorLoc<ELF32BE>(Loc);
  case ELF64LEKind:
    return getErrorLoc<ELF64LE>(Loc);
  case ELF64BEKind:
    return getErrorLoc<ELF64BE>(Loc);
  default:
    llvm_unreachable("unknown ELF type");
  }
}

template void elf::writeResult<ELF32LE>();
template void elf::writeResult<ELF32BE>();
template void elf::writeResult<ELF64LE>();
template void elf::writeResult<ELF64BE>();

template struct elf::PhdrEntry<ELF32LE>;
template struct elf::PhdrEntry<ELF32BE>;
template struct elf::PhdrEntry<ELF64LE>;
template struct elf::PhdrEntry<ELF64BE>;

template bool elf::isRelroSection<ELF32LE>(const OutputSectionBase *);
template bool elf::isRelroSection<ELF32BE>(const OutputSectionBase *);
template bool elf::isRelroSection<ELF64LE>(const OutputSectionBase *);
template bool elf::isRelroSection<ELF64BE>(const OutputSectionBase *);

template void elf::reportDiscarded<ELF32LE>(InputSectionBase<ELF32LE> *);
template void elf::reportDiscarded<ELF32BE>(InputSectionBase<ELF32BE> *);
template void elf::reportDiscarded<ELF64LE>(InputSectionBase<ELF64LE> *);
template void elf::reportDiscarded<ELF64BE>(InputSectionBase<ELF64BE> *);
