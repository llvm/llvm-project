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
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Target.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

namespace {
// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFFile<ELFT>::Elf_Phdr Elf_Phdr;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;
  Writer(SymbolTable<ELFT> &S) : Symtab(S) {}
  void run();

private:
  void copyLocalSymbols();
  void createSections();
  template <bool isRela>
  void scanRelocs(const InputSection<ELFT> &C,
                  iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels);
  void scanRelocs(const InputSection<ELFT> &C);
  void assignAddresses();
  void openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();
  bool needsInterpSection() const {
    return !Symtab.getSharedFiles().empty() && !Config->DynamicLinker.empty();
  }
  bool isOutputDynamic() const {
    return !Symtab.getSharedFiles().empty() || Config->Shared;
  }
  uintX_t getEntryAddr() const;
  int getPhdrsNum() const;

  OutputSection<ELFT> *getBSS();
  void addCommonSymbols(std::vector<DefinedCommon<ELFT> *> &Syms);
  void addSharedCopySymbols(std::vector<SharedSymbol<ELFT> *> &Syms);

  std::unique_ptr<llvm::FileOutputBuffer> Buffer;

  SpecificBumpPtrAllocator<OutputSection<ELFT>> SecAlloc;
  SpecificBumpPtrAllocator<MergeOutputSection<ELFT>> MSecAlloc;
  BumpPtrAllocator Alloc;
  std::vector<OutputSectionBase<ELFT> *> OutputSections;
  unsigned getNumSections() const { return OutputSections.size() + 1; }

  void addStartStopSymbols(OutputSectionBase<ELFT> *Sec);
  void setPhdr(Elf_Phdr *PH, uint32_t Type, uint32_t Flags, uintX_t FileOff,
               uintX_t VA, uintX_t Size, uintX_t Align);
  void copyPhdr(Elf_Phdr *PH, OutputSectionBase<ELFT> *From);

  SymbolTable<ELFT> &Symtab;
  std::vector<Elf_Phdr> Phdrs;

  uintX_t FileSize;
  uintX_t SectionHeaderOff;
};
} // anonymous namespace

template <class ELFT> void lld::elf2::writeResult(SymbolTable<ELFT> *Symtab) {
  // Initialize output sections that are handled by Writer specially.
  // Don't reorder because the order of initialization matters.
  InterpSection<ELFT> Interp;
  Out<ELFT>::Interp = &Interp;
  StringTableSection<ELFT> ShStrTab(".shstrtab", false);
  Out<ELFT>::ShStrTab = &ShStrTab;
  StringTableSection<ELFT> StrTab(".strtab", false);
  if (!Config->StripAll)
    Out<ELFT>::StrTab = &StrTab;
  StringTableSection<ELFT> DynStrTab(".dynstr", true);
  Out<ELFT>::DynStrTab = &DynStrTab;
  GotSection<ELFT> Got;
  Out<ELFT>::Got = &Got;
  GotPltSection<ELFT> GotPlt;
  if (Target->supportsLazyRelocations())
    Out<ELFT>::GotPlt = &GotPlt;
  PltSection<ELFT> Plt;
  Out<ELFT>::Plt = &Plt;
  std::unique_ptr<SymbolTableSection<ELFT>> SymTab;
  if (!Config->StripAll) {
    SymTab.reset(new SymbolTableSection<ELFT>(*Symtab, *Out<ELFT>::StrTab));
    Out<ELFT>::SymTab = SymTab.get();
  }
  SymbolTableSection<ELFT> DynSymTab(*Symtab, *Out<ELFT>::DynStrTab);
  Out<ELFT>::DynSymTab = &DynSymTab;
  HashTableSection<ELFT> HashTab;
  if (Config->SysvHash)
    Out<ELFT>::HashTab = &HashTab;
  GnuHashTableSection<ELFT> GnuHashTab;
  if (Config->GnuHash)
    Out<ELFT>::GnuHashTab = &GnuHashTab;
  bool IsRela = Symtab->shouldUseRela();
  RelocationSection<ELFT> RelaDyn(IsRela ? ".rela.dyn" : ".rel.dyn", IsRela);
  Out<ELFT>::RelaDyn = &RelaDyn;
  RelocationSection<ELFT> RelaPlt(IsRela ? ".rela.plt" : ".rel.plt", IsRela);
  if (Target->supportsLazyRelocations())
    Out<ELFT>::RelaPlt = &RelaPlt;
  DynamicSection<ELFT> Dynamic(*Symtab);
  Out<ELFT>::Dynamic = &Dynamic;

  Writer<ELFT>(*Symtab).run();
}

// The main function of the writer.
template <class ELFT> void Writer<ELFT>::run() {
  if (!Config->DiscardAll)
    copyLocalSymbols();
  createSections();
  assignAddresses();
  openFile(Config->OutputFile);
  writeHeader();
  writeSections();
  error(Buffer->commit());
}

namespace {
template <bool Is64Bits> struct SectionKey {
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  StringRef Name;
  uint32_t Type;
  uintX_t Flags;
  uintX_t EntSize;
};
}
namespace llvm {
template <bool Is64Bits> struct DenseMapInfo<SectionKey<Is64Bits>> {
  static SectionKey<Is64Bits> getEmptyKey() {
    return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getEmptyKey(), 0, 0,
                                0};
  }
  static SectionKey<Is64Bits> getTombstoneKey() {
    return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getTombstoneKey(), 0,
                                0, 0};
  }
  static unsigned getHashValue(const SectionKey<Is64Bits> &Val) {
    return hash_combine(Val.Name, Val.Type, Val.Flags, Val.EntSize);
  }
  static bool isEqual(const SectionKey<Is64Bits> &LHS,
                      const SectionKey<Is64Bits> &RHS) {
    return DenseMapInfo<StringRef>::isEqual(LHS.Name, RHS.Name) &&
           LHS.Type == RHS.Type && LHS.Flags == RHS.Flags &&
           LHS.EntSize == RHS.EntSize;
  }
};
}

// The reason we have to do this early scan is as follows
// * To mmap the output file, we need to know the size
// * For that, we need to know how many dynamic relocs we will have.
// It might be possible to avoid this by outputting the file with write:
// * Write the allocated output sections, computing addresses.
// * Apply relocations, recording which ones require a dynamic reloc.
// * Write the dynamic relocations.
// * Write the rest of the file.
template <class ELFT>
template <bool isRela>
void Writer<ELFT>::scanRelocs(
    const InputSection<ELFT> &C,
    iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  const ObjectFile<ELFT> &File = *C.getFile();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    SymbolBody *Body = File.getSymbolBody(SymIndex);
    uint32_t Type = RI.getType(Config->Mips64EL);

    // Set "used" bit for --as-needed.
    if (Body && Body->isUndefined() && !Body->isWeak())
      if (auto *S = dyn_cast<SharedSymbol<ELFT>>(Body->repl()))
        S->File->IsUsed = true;

    if (Body)
      Body = Body->repl();
    bool NeedsGot = false;
    bool NeedsPlt = false;
    if (Body) {
      if (auto *E = dyn_cast<SharedSymbol<ELFT>>(Body)) {
        if (E->needsCopy())
          continue;
        if (Target->relocNeedsCopy(Type, *Body))
          E->OffsetInBSS = 0;
      }
      NeedsPlt = Target->relocNeedsPlt(Type, *Body);
      if (NeedsPlt) {
        if (Body->isInPlt())
          continue;
        Out<ELFT>::Plt->addEntry(Body);
      }
      NeedsGot = Target->relocNeedsGot(Type, *Body);
      if (NeedsGot) {
        if (NeedsPlt && Target->supportsLazyRelocations()) {
          Out<ELFT>::GotPlt->addEntry(Body);
        } else {
          if (Body->isInGot())
            continue;
          Out<ELFT>::Got->addEntry(Body);
        }
      }
    }

    if (Config->EMachine == EM_MIPS && NeedsGot) {
      // MIPS ABI has special rules to process GOT entries
      // and doesn't require relocation entries for them.
      // See "Global Offset Table" in Chapter 5 in the following document
      // for detailed description:
      // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
      Body->setUsedInDynamicReloc();
      continue;
    }
    bool CBP = canBePreempted(Body, NeedsGot);
    if (!CBP && (!Config->Shared || Target->isRelRelative(Type)))
      continue;
    if (CBP)
      Body->setUsedInDynamicReloc();
    if (NeedsPlt && Target->supportsLazyRelocations())
      Out<ELFT>::RelaPlt->addReloc({C, RI});
    else
      Out<ELFT>::RelaDyn->addReloc({C, RI});
  }
}

template <class ELFT>
void Writer<ELFT>::scanRelocs(const InputSection<ELFT> &C) {
  ObjectFile<ELFT> *File = C.getFile();
  ELFFile<ELFT> &EObj = File->getObj();

  if (!(C.getSectionHdr()->sh_flags & SHF_ALLOC))
    return;

  for (const Elf_Shdr *RelSec : C.RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      scanRelocs(C, EObj.relas(RelSec));
    else
      scanRelocs(C, EObj.rels(RelSec));
  }
}

template <class ELFT>
static void reportUndefined(const SymbolTable<ELFT> &S, const SymbolBody &Sym) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  if (Config->Shared && !Config->NoUndefined)
    return;

  const Elf_Sym &SymE = cast<ELFSymbolBody<ELFT>>(Sym).Sym;
  ELFFileBase<ELFT> *SymFile = nullptr;

  for (const std::unique_ptr<ObjectFile<ELFT>> &File : S.getObjectFiles()) {
    Elf_Sym_Range Syms = File->getObj().symbols(File->getSymbolTable());
    if (&SymE > Syms.begin() && &SymE < Syms.end())
      SymFile = File.get();
  }

  std::string Message = "undefined symbol: " + Sym.getName().str();
  if (SymFile)
    Message += " in " + SymFile->getName().str();
  if (Config->NoInhibitExec)
    warning(Message);
  else
    error(Message);
}

// Local symbols are not in the linker's symbol table. This function scans
// each object file's symbol table to copy local symbols to the output.
template <class ELFT> void Writer<ELFT>::copyLocalSymbols() {
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : Symtab.getObjectFiles()) {
    for (const Elf_Sym &Sym : F->getLocalSymbols()) {
      ErrorOr<StringRef> SymNameOrErr = Sym.getName(F->getStringTable());
      error(SymNameOrErr);
      StringRef SymName = *SymNameOrErr;
      if (!shouldKeepInSymtab<ELFT>(*F, SymName, Sym))
        continue;
      if (Out<ELFT>::SymTab)
        Out<ELFT>::SymTab->addLocalSymbol(SymName);
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

// Output section ordering is determined by this function.
template <class ELFT>
static bool compareOutputSections(OutputSectionBase<ELFT> *A,
                                  OutputSectionBase<ELFT> *B) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;

  uintX_t AFlags = A->getFlags();
  uintX_t BFlags = B->getFlags();

  // Allocatable sections go first to reduce the total PT_LOAD size and
  // so debug info doesn't change addresses in actual code.
  bool AIsAlloc = AFlags & SHF_ALLOC;
  bool BIsAlloc = BFlags & SHF_ALLOC;
  if (AIsAlloc != BIsAlloc)
    return AIsAlloc;

  // We don't have any special requirements for the relative order of
  // two non allocatable sections.
  if (!AIsAlloc)
    return false;

  // We want the read only sections first so that they go in the PT_LOAD
  // covering the program headers at the start of the file.
  bool AIsWritable = AFlags & SHF_WRITE;
  bool BIsWritable = BFlags & SHF_WRITE;
  if (AIsWritable != BIsWritable)
    return BIsWritable;

  // For a corresponding reason, put non exec sections first (the program
  // header PT_LOAD is not executable).
  bool AIsExec = AFlags & SHF_EXECINSTR;
  bool BIsExec = BFlags & SHF_EXECINSTR;
  if (AIsExec != BIsExec)
    return BIsExec;

  // If we got here we know that both A and B are in the same PT_LOAD.

  // The TLS initialization block needs to be a single contiguous block in a R/W
  // PT_LOAD, so stick TLS sections directly before R/W sections. The TLS NOBITS
  // sections are placed here as they don't take up virtual address space in the
  // PT_LOAD.
  bool AIsTLS = AFlags & SHF_TLS;
  bool BIsTLS = BFlags & SHF_TLS;
  if (AIsTLS != BIsTLS)
    return AIsTLS;

  // The next requirement we have is to put nobits sections last. The
  // reason is that the only thing the dynamic linker will see about
  // them is a p_memsz that is larger than p_filesz. Seeing that it
  // zeros the end of the PT_LOAD, so that has to correspond to the
  // nobits sections.
  bool AIsNoBits = A->getType() == SHT_NOBITS;
  bool BIsNoBits = B->getType() == SHT_NOBITS;
  if (AIsNoBits != BIsNoBits)
    return BIsNoBits;

  // Some architectures have additional ordering restrictions for sections
  // within the same PT_LOAD.
  if (Config->EMachine == EM_PPC64)
    return getPPC64SectionRank(A->getName()) <
           getPPC64SectionRank(B->getName());

  return false;
}

template <class ELFT> OutputSection<ELFT> *Writer<ELFT>::getBSS() {
  if (!Out<ELFT>::Bss) {
    Out<ELFT>::Bss = new (SecAlloc.Allocate())
        OutputSection<ELFT>(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE);
    OutputSections.push_back(Out<ELFT>::Bss);
  }
  return Out<ELFT>::Bss;
}

// Until this function is called, common symbols do not belong to any section.
// This function adds them to end of BSS section.
template <class ELFT>
void Writer<ELFT>::addCommonSymbols(std::vector<DefinedCommon<ELFT> *> &Syms) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;

  if (Syms.empty())
    return;

  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::stable_sort(
    Syms.begin(), Syms.end(),
    [](const DefinedCommon<ELFT> *A, const DefinedCommon<ELFT> *B) {
      return A->MaxAlignment > B->MaxAlignment;
    });

  uintX_t Off = getBSS()->getSize();
  for (DefinedCommon<ELFT> *C : Syms) {
    const Elf_Sym &Sym = C->Sym;
    uintX_t Align = C->MaxAlignment;
    Off = RoundUpToAlignment(Off, Align);
    C->OffsetInBSS = Off;
    Off += Sym.st_size;
  }

  Out<ELFT>::Bss->setSize(Off);
}

template <class ELFT>
void Writer<ELFT>::addSharedCopySymbols(
    std::vector<SharedSymbol<ELFT> *> &Syms) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

  if (Syms.empty())
    return;

  uintX_t Off = getBSS()->getSize();
  for (SharedSymbol<ELFT> *C : Syms) {
    const Elf_Sym &Sym = C->Sym;
    const Elf_Shdr *Sec = C->File->getSection(Sym);
    uintX_t SecAlign = Sec->sh_addralign;
    uintX_t Align = Sym.st_value % SecAlign;
    if (Align == 0)
      Align = SecAlign;
    Out<ELFT>::Bss->updateAlign(Align);
    Off = RoundUpToAlignment(Off, Align);
    C->OffsetInBSS = Off;
    Off += Sym.st_size;
  }
  Out<ELFT>::Bss->setSize(Off);
}

static StringRef getOutputName(StringRef S) {
  if (S.startswith(".text."))
    return ".text";
  if (S.startswith(".rodata."))
    return ".rodata";
  if (S.startswith(".data."))
    return ".data";
  if (S.startswith(".bss."))
    return ".bss";
  return S;
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  // .interp needs to be on the first page in the output file.
  if (needsInterpSection())
    OutputSections.push_back(Out<ELFT>::Interp);

  SmallDenseMap<SectionKey<ELFT::Is64Bits>, OutputSectionBase<ELFT> *> Map;

  std::vector<OutputSectionBase<ELFT> *> RegularSections;

  for (const std::unique_ptr<ObjectFile<ELFT>> &F : Symtab.getObjectFiles()) {
    for (InputSectionBase<ELFT> *C : F->getSections()) {
      if (!C || !C->isLive() || C == &InputSection<ELFT>::Discarded)
        continue;
      const Elf_Shdr *H = C->getSectionHdr();
      uintX_t OutFlags = H->sh_flags & ~SHF_GROUP;
      // For SHF_MERGE we create different output sections for each sh_entsize.
      // This makes each output section simple and keeps a single level
      // mapping from input to output.
      auto *IS = dyn_cast<InputSection<ELFT>>(C);
      uintX_t EntSize = IS ? 0 : H->sh_entsize;
      SectionKey<ELFT::Is64Bits> Key{getOutputName(C->getSectionName()),
                                     H->sh_type, OutFlags, EntSize};
      OutputSectionBase<ELFT> *&Sec = Map[Key];
      if (!Sec) {
        if (IS)
          Sec = new (SecAlloc.Allocate())
              OutputSection<ELFT>(Key.Name, Key.Type, Key.Flags);
        else
          Sec = new (MSecAlloc.Allocate())
              MergeOutputSection<ELFT>(Key.Name, Key.Type, Key.Flags);
        OutputSections.push_back(Sec);
        RegularSections.push_back(Sec);
      }
      if (IS)
        static_cast<OutputSection<ELFT> *>(Sec)->addSection(IS);
      else
        static_cast<MergeOutputSection<ELFT> *>(Sec)
            ->addSection(cast<MergeInputSection<ELFT>>(C));
    }
  }

  Out<ELFT>::Bss = static_cast<OutputSection<ELFT> *>(
      Map[{".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0}]);

  Out<ELFT>::Dynamic->PreInitArraySec = Map.lookup(
      {".preinit_array", SHT_PREINIT_ARRAY, SHF_WRITE | SHF_ALLOC, 0});
  Out<ELFT>::Dynamic->InitArraySec =
      Map.lookup({".init_array", SHT_INIT_ARRAY, SHF_WRITE | SHF_ALLOC, 0});
  Out<ELFT>::Dynamic->FiniArraySec =
      Map.lookup({".fini_array", SHT_FINI_ARRAY, SHF_WRITE | SHF_ALLOC, 0});

  auto AddStartEnd = [&](StringRef Start, StringRef End,
                         OutputSectionBase<ELFT> *OS) {
    if (OS) {
      Symtab.addSyntheticSym(Start, *OS, 0);
      Symtab.addSyntheticSym(End, *OS, OS->getSize());
    } else {
      Symtab.addIgnoredSym(Start);
      Symtab.addIgnoredSym(End);
    }
  };

  AddStartEnd("__preinit_array_start", "__preinit_array_end",
              Out<ELFT>::Dynamic->PreInitArraySec);
  AddStartEnd("__init_array_start", "__init_array_end",
              Out<ELFT>::Dynamic->InitArraySec);
  AddStartEnd("__fini_array_start", "__fini_array_end",
              Out<ELFT>::Dynamic->FiniArraySec);

  for (OutputSectionBase<ELFT> *Sec : RegularSections)
    addStartStopSymbols(Sec);

  // __tls_get_addr is defined by the dynamic linker for dynamic ELFs. For
  // static linking the linker is required to optimize away any references to
  // __tls_get_addr, so it's not defined anywhere. Create a hidden definition
  // to avoid the undefined symbol error.
  if (!isOutputDynamic())
    Symtab.addIgnoredSym("__tls_get_addr");

  // Scan relocations. This must be done after every symbol is declared so that
  // we can correctly decide if a dynamic relocation is needed.
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : Symtab.getObjectFiles())
    for (InputSectionBase<ELFT> *B : F->getSections())
      if (auto *S = dyn_cast_or_null<InputSection<ELFT>>(B))
        if (S != &InputSection<ELFT>::Discarded)
          if (S->isLive())
            scanRelocs(*S);

  std::vector<DefinedCommon<ELFT> *> CommonSymbols;
  std::vector<SharedSymbol<ELFT> *> SharedCopySymbols;
  for (auto &P : Symtab.getSymbols()) {
    SymbolBody *Body = P.second->Body;
    if (auto *U = dyn_cast<Undefined<ELFT>>(Body))
      if (!U->isWeak() && !U->canKeepUndefined())
        reportUndefined<ELFT>(Symtab, *Body);

    if (auto *C = dyn_cast<DefinedCommon<ELFT>>(Body))
      CommonSymbols.push_back(C);
    if (auto *SC = dyn_cast<SharedSymbol<ELFT>>(Body))
      if (SC->needsCopy())
        SharedCopySymbols.push_back(SC);

    if (!includeInSymtab<ELFT>(*Body))
      continue;
    if (Out<ELFT>::SymTab)
      Out<ELFT>::SymTab->addSymbol(Body);

    if (isOutputDynamic() && includeInDynamicSymtab(*Body))
      Out<ELFT>::DynSymTab->addSymbol(Body);
  }
  addCommonSymbols(CommonSymbols);
  addSharedCopySymbols(SharedCopySymbols);

  // This order is not the same as the final output order
  // because we sort the sections using their attributes below.
  if (Out<ELFT>::SymTab)
    OutputSections.push_back(Out<ELFT>::SymTab);
  OutputSections.push_back(Out<ELFT>::ShStrTab);
  if (Out<ELFT>::StrTab)
    OutputSections.push_back(Out<ELFT>::StrTab);
  if (isOutputDynamic()) {
    OutputSections.push_back(Out<ELFT>::DynSymTab);
    if (Out<ELFT>::GnuHashTab)
      OutputSections.push_back(Out<ELFT>::GnuHashTab);
    if (Out<ELFT>::HashTab)
      OutputSections.push_back(Out<ELFT>::HashTab);
    OutputSections.push_back(Out<ELFT>::Dynamic);
    OutputSections.push_back(Out<ELFT>::DynStrTab);
    if (Out<ELFT>::RelaDyn->hasRelocs())
      OutputSections.push_back(Out<ELFT>::RelaDyn);
    if (Out<ELFT>::RelaPlt && Out<ELFT>::RelaPlt->hasRelocs())
      OutputSections.push_back(Out<ELFT>::RelaPlt);
  }
  if (!Out<ELFT>::Got->empty())
    OutputSections.push_back(Out<ELFT>::Got);
  if (Out<ELFT>::GotPlt && !Out<ELFT>::GotPlt->empty())
    OutputSections.push_back(Out<ELFT>::GotPlt);
  if (!Out<ELFT>::Plt->empty())
    OutputSections.push_back(Out<ELFT>::Plt);

  std::stable_sort(OutputSections.begin(), OutputSections.end(),
                   compareOutputSections<ELFT>);

  for (unsigned I = 0, N = OutputSections.size(); I < N; ++I)
    OutputSections[I]->SectionIndex = I + 1;

  for (OutputSectionBase<ELFT> *Sec : OutputSections)
    Out<ELFT>::ShStrTab->add(Sec->getName());

  // Finalizers fix each section's size.
  // .dynamic section's finalizer may add strings to .dynstr,
  // so finalize that early.
  // Likewise, .dynsym is finalized early since that may fill up .gnu.hash.
  Out<ELFT>::Dynamic->finalize();
  if (isOutputDynamic())
    Out<ELFT>::DynSymTab->finalize();

  // Fill other section headers.
  for (OutputSectionBase<ELFT> *Sec : OutputSections)
    Sec->finalize();

  // If we have a .opd section (used under PPC64 for function descriptors),
  // store a pointer to it here so that we can use it later when processing
  // relocations.
  Out<ELFT>::Opd = Map.lookup({".opd", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC, 0});
}

static bool isAlpha(char C) {
  return ('a' <= C && C <= 'z') || ('A' <= C && C <= 'Z') || C == '_';
}

static bool isAlnum(char C) { return isAlpha(C) || ('0' <= C && C <= '9'); }

// Returns true if S is valid as a C language identifier.
static bool isValidCIdentifier(StringRef S) {
  if (S.empty() || !isAlpha(S[0]))
    return false;
  return std::all_of(S.begin() + 1, S.end(), isAlnum);
}

// If a section name is valid as a C identifier (which is rare because of
// the leading '.'), linkers are expected to define __start_<secname> and
// __stop_<secname> symbols. They are at beginning and end of the section,
// respectively. This is not requested by the ELF standard, but GNU ld and
// gold provide the feature, and used by many programs.
template <class ELFT>
void Writer<ELFT>::addStartStopSymbols(OutputSectionBase<ELFT> *Sec) {
  StringRef S = Sec->getName();
  if (!isValidCIdentifier(S))
    return;
  StringSaver Saver(Alloc);
  StringRef Start = Saver.save("__start_" + S);
  StringRef Stop = Saver.save("__stop_" + S);
  if (Symtab.isUndefined(Start))
    Symtab.addSyntheticSym(Start, *Sec, 0);
  if (Symtab.isUndefined(Stop))
    Symtab.addSyntheticSym(Stop, *Sec, Sec->getSize());
}

template <class ELFT> static bool needsPhdr(OutputSectionBase<ELFT> *Sec) {
  return Sec->getFlags() & SHF_ALLOC;
}

static uint32_t toPhdrFlags(uint64_t Flags) {
  uint32_t Ret = PF_R;
  if (Flags & SHF_WRITE)
    Ret |= PF_W;
  if (Flags & SHF_EXECINSTR)
    Ret |= PF_X;
  return Ret;
}

// Visits all sections to create PHDRs and to assign incremental,
// non-overlapping addresses to output sections.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  uintX_t VA = Target->getVAStart() + sizeof(Elf_Ehdr);
  uintX_t FileOff = sizeof(Elf_Ehdr);

  // Calculate and reserve the space for the program header first so that
  // the first section can start right after the program header.
  Phdrs.resize(getPhdrsNum());
  size_t PhdrSize = sizeof(Elf_Phdr) * Phdrs.size();

  // The first phdr entry is PT_PHDR which describes the program header itself.
  setPhdr(&Phdrs[0], PT_PHDR, PF_R, FileOff, VA, PhdrSize, /*Align=*/8);
  FileOff += PhdrSize;
  VA += PhdrSize;

  // PT_INTERP must be the second entry if exists.
  int PhdrIdx = 0;
  Elf_Phdr *Interp = nullptr;
  if (needsInterpSection())
    Interp = &Phdrs[++PhdrIdx];

  // Add the first PT_LOAD segment for regular output sections.
  setPhdr(&Phdrs[++PhdrIdx], PT_LOAD, PF_R, 0, Target->getVAStart(), FileOff,
          Target->getPageSize());

  Elf_Phdr TlsPhdr{};
  uintX_t ThreadBSSOffset = 0;
  // Create phdrs as we assign VAs and file offsets to all output sections.
  for (OutputSectionBase<ELFT> *Sec : OutputSections) {
    if (needsPhdr<ELFT>(Sec)) {
      uintX_t Flags = toPhdrFlags(Sec->getFlags());
      if (Phdrs[PhdrIdx].p_flags != Flags) {
        // Flags changed. Create a new PT_LOAD.
        VA = RoundUpToAlignment(VA, Target->getPageSize());
        FileOff = RoundUpToAlignment(FileOff, Target->getPageSize());
        Elf_Phdr *PH = &Phdrs[++PhdrIdx];
        setPhdr(PH, PT_LOAD, Flags, FileOff, VA, 0, Target->getPageSize());
      }

      if (Sec->getFlags() & SHF_TLS) {
        if (!TlsPhdr.p_vaddr)
          setPhdr(&TlsPhdr, PT_TLS, PF_R, FileOff, VA, 0, Sec->getAlign());
        if (Sec->getType() != SHT_NOBITS)
          VA = RoundUpToAlignment(VA, Sec->getAlign());
        uintX_t TVA = RoundUpToAlignment(VA + ThreadBSSOffset, Sec->getAlign());
        Sec->setVA(TVA);
        TlsPhdr.p_memsz += Sec->getSize();
        if (Sec->getType() == SHT_NOBITS) {
          ThreadBSSOffset = TVA - VA + Sec->getSize();
        } else {
          TlsPhdr.p_filesz += Sec->getSize();
          VA += Sec->getSize();
        }
        TlsPhdr.p_align = std::max<uintX_t>(TlsPhdr.p_align, Sec->getAlign());
      } else {
        VA = RoundUpToAlignment(VA, Sec->getAlign());
        Sec->setVA(VA);
        VA += Sec->getSize();
      }
    }

    FileOff = RoundUpToAlignment(FileOff, Sec->getAlign());
    Sec->setFileOffset(FileOff);
    if (Sec->getType() != SHT_NOBITS)
      FileOff += Sec->getSize();
    if (needsPhdr<ELFT>(Sec)) {
      Elf_Phdr *Cur = &Phdrs[PhdrIdx];
      Cur->p_filesz = FileOff - Cur->p_offset;
      Cur->p_memsz = VA - Cur->p_vaddr;
    }
  }

  if (TlsPhdr.p_vaddr) {
    // The TLS pointer goes after PT_TLS. At least glibc will align it,
    // so round up the size to make sure the offsets are correct.
    TlsPhdr.p_memsz = RoundUpToAlignment(TlsPhdr.p_memsz, TlsPhdr.p_align);
    Phdrs[++PhdrIdx] = TlsPhdr;
    Out<ELFT>::TlsPhdr = &Phdrs[PhdrIdx];
  }

  // Add an entry for .dynamic.
  if (isOutputDynamic()) {
    Elf_Phdr *PH = &Phdrs[++PhdrIdx];
    PH->p_type = PT_DYNAMIC;
    copyPhdr(PH, Out<ELFT>::Dynamic);
  }

  // Fix up PT_INTERP as we now know the address of .interp section.
  if (Interp) {
    Interp->p_type = PT_INTERP;
    copyPhdr(Interp, Out<ELFT>::Interp);
  }

  // Add space for section headers.
  SectionHeaderOff = RoundUpToAlignment(FileOff, ELFT::Is64Bits ? 8 : 4);
  FileSize = SectionHeaderOff + getNumSections() * sizeof(Elf_Shdr);

  // Update MIPS _gp absolute symbol so that it points to the static data.
  if (Config->EMachine == EM_MIPS)
    DefinedAbsolute<ELFT>::MipsGp.st_value = getMipsGpAddr<ELFT>();
}

// Returns the number of PHDR entries.
template <class ELFT> int Writer<ELFT>::getPhdrsNum() const {
  bool Tls = false;
  int I = 2; // 2 for PT_PHDR and the first PT_LOAD
  if (needsInterpSection())
    ++I;
  if (isOutputDynamic())
    ++I;
  uintX_t Last = PF_R;
  for (OutputSectionBase<ELFT> *Sec : OutputSections) {
    if (!needsPhdr<ELFT>(Sec))
      continue;
    if (Sec->getFlags() & SHF_TLS)
      Tls = true;
    uintX_t Flags = toPhdrFlags(Sec->getFlags());
    if (Last != Flags) {
      Last = Flags;
      ++I;
    }
  }
  if (Tls)
    ++I;
  return I;
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  memcpy(Buf, "\177ELF", 4);

  // Write the ELF header.
  auto *EHdr = reinterpret_cast<Elf_Ehdr *>(Buf);
  EHdr->e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  EHdr->e_ident[EI_DATA] = ELFT::TargetEndianness == llvm::support::little
                               ? ELFDATA2LSB
                               : ELFDATA2MSB;
  EHdr->e_ident[EI_VERSION] = EV_CURRENT;

  auto &FirstObj = cast<ELFFileBase<ELFT>>(*Config->FirstElf);
  EHdr->e_ident[EI_OSABI] = FirstObj.getOSABI();

  EHdr->e_type = Config->Shared ? ET_DYN : ET_EXEC;
  EHdr->e_machine = FirstObj.getEMachine();
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = getEntryAddr();
  EHdr->e_phoff = sizeof(Elf_Ehdr);
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_ehsize = sizeof(Elf_Ehdr);
  EHdr->e_phentsize = sizeof(Elf_Phdr);
  EHdr->e_phnum = Phdrs.size();
  EHdr->e_shentsize = sizeof(Elf_Shdr);
  EHdr->e_shnum = getNumSections();
  EHdr->e_shstrndx = Out<ELFT>::ShStrTab->SectionIndex;

  // Write the program header table.
  memcpy(Buf + EHdr->e_phoff, &Phdrs[0], Phdrs.size() * sizeof(Phdrs[0]));

  // Write the section header table. Note that the first table entry is null.
  auto SHdrs = reinterpret_cast<Elf_Shdr *>(Buf + EHdr->e_shoff);
  for (OutputSectionBase<ELFT> *Sec : OutputSections)
    Sec->writeHeaderTo(++SHdrs);
}

template <class ELFT> void Writer<ELFT>::openFile(StringRef Path) {
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Path, FileSize, FileOutputBuffer::F_executable);
  error(BufferOrErr, Twine("failed to open ") + Path);
  Buffer = std::move(*BufferOrErr);
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();

  // PPC64 needs to process relocations in the .opd section before processing
  // relocations in code-containing sections.
  if (OutputSectionBase<ELFT> *Sec = Out<ELFT>::Opd) {
    Out<ELFT>::OpdBuf = Buf + Sec->getFileOff();
    Sec->writeTo(Buf + Sec->getFileOff());
  }

  for (OutputSectionBase<ELFT> *Sec : OutputSections)
    if (Sec != Out<ELFT>::Opd)
      Sec->writeTo(Buf + Sec->getFileOff());
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t Writer<ELFT>::getEntryAddr() const {
  if (Config->EntrySym) {
    if (auto *E = dyn_cast<ELFSymbolBody<ELFT>>(Config->EntrySym->repl()))
      return getSymVA<ELFT>(*E);
    return 0;
  }
  if (Config->EntryAddr != uint64_t(-1))
    return Config->EntryAddr;
  return 0;
}

template <class ELFT>
void Writer<ELFT>::setPhdr(Elf_Phdr *PH, uint32_t Type, uint32_t Flags,
                           uintX_t FileOff, uintX_t VA, uintX_t Size,
                           uintX_t Align) {
  PH->p_type = Type;
  PH->p_flags = Flags;
  PH->p_offset = FileOff;
  PH->p_vaddr = VA;
  PH->p_paddr = VA;
  PH->p_filesz = Size;
  PH->p_memsz = Size;
  PH->p_align = Align;
}

template <class ELFT>
void Writer<ELFT>::copyPhdr(Elf_Phdr *PH, OutputSectionBase<ELFT> *From) {
  PH->p_flags = toPhdrFlags(From->getFlags());
  PH->p_offset = From->getFileOff();
  PH->p_vaddr = From->getVA();
  PH->p_paddr = From->getVA();
  PH->p_filesz = From->getSize();
  PH->p_memsz = From->getSize();
  PH->p_align = From->getAlign();
}

template void lld::elf2::writeResult<ELF32LE>(SymbolTable<ELF32LE> *Symtab);
template void lld::elf2::writeResult<ELF32BE>(SymbolTable<ELF32BE> *Symtab);
template void lld::elf2::writeResult<ELF64LE>(SymbolTable<ELF64LE> *Symtab);
template void lld::elf2::writeResult<ELF64BE>(SymbolTable<ELF64BE> *Symtab);
