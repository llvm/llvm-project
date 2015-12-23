//===- OutputSections.cpp -------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "Config.h"
#include "SymbolTable.h"
#include "Target.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

bool lld::elf2::HasGotOffRel = false;

template <class ELFT>
OutputSectionBase<ELFT>::OutputSectionBase(StringRef Name, uint32_t sh_type,
                                           uintX_t sh_flags)
    : Name(Name) {
  memset(&Header, 0, sizeof(Elf_Shdr));
  Header.sh_type = sh_type;
  Header.sh_flags = sh_flags;
}

template <class ELFT>
GotPltSection<ELFT>::GotPltSection()
    : OutputSectionBase<ELFT>(".got.plt", llvm::ELF::SHT_PROGBITS,
                              llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE) {
  this->Header.sh_addralign = sizeof(uintX_t);
}

template <class ELFT> void GotPltSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->GotPltIndex = Target->getGotPltHeaderEntriesNum() + Entries.size();
  Entries.push_back(Sym);
}

template <class ELFT> bool GotPltSection<ELFT>::empty() const {
  return Entries.empty();
}

template <class ELFT>
typename GotPltSection<ELFT>::uintX_t
GotPltSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.GotPltIndex * sizeof(uintX_t);
}

template <class ELFT> void GotPltSection<ELFT>::finalize() {
  this->Header.sh_size =
      (Target->getGotPltHeaderEntriesNum() + Entries.size()) * sizeof(uintX_t);
}

template <class ELFT> void GotPltSection<ELFT>::writeTo(uint8_t *Buf) {
  Target->writeGotPltHeaderEntries(Buf);
  Buf += Target->getGotPltHeaderEntriesNum() * sizeof(uintX_t);
  for (const SymbolBody *B : Entries) {
    Target->writeGotPltEntry(Buf, Out<ELFT>::Plt->getEntryAddr(*B));
    Buf += sizeof(uintX_t);
  }
}

template <class ELFT>
GotSection<ELFT>::GotSection()
    : OutputSectionBase<ELFT>(".got", llvm::ELF::SHT_PROGBITS,
                              llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE) {
  if (Config->EMachine == EM_MIPS)
    this->Header.sh_flags |= llvm::ELF::SHF_MIPS_GPREL;
  this->Header.sh_addralign = sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->GotIndex = Target->getGotHeaderEntriesNum() + Entries.size();
  Entries.push_back(Sym);
}

template <class ELFT> bool GotSection<ELFT>::addDynTlsEntry(SymbolBody *Sym) {
  if (Sym->hasGlobalDynIndex())
    return false;
  Sym->GlobalDynIndex = Target->getGotHeaderEntriesNum() + Entries.size();
  // Global Dynamic TLS entries take two GOT slots.
  Entries.push_back(Sym);
  Entries.push_back(nullptr);
  return true;
}

template <class ELFT> bool GotSection<ELFT>::addCurrentModuleTlsIndex() {
  if (LocalTlsIndexOff != uint32_t(-1))
    return false;
  Entries.push_back(nullptr);
  Entries.push_back(nullptr);
  LocalTlsIndexOff = (Entries.size() - 2) * sizeof(uintX_t);
  return true;
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + B.GotIndex * sizeof(uintX_t);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getGlobalDynAddr(const SymbolBody &B) const {
  return this->getVA() + B.GlobalDynIndex * sizeof(uintX_t);
}

template <class ELFT>
const SymbolBody *GotSection<ELFT>::getMipsFirstGlobalEntry() const {
  return Entries.empty() ? nullptr : Entries.front();
}

template <class ELFT>
unsigned GotSection<ELFT>::getMipsLocalEntriesNum() const {
  // TODO: Update when the suppoort of GOT entries for local symbols is added.
  return Target->getGotHeaderEntriesNum();
}

template <class ELFT> void GotSection<ELFT>::finalize() {
  this->Header.sh_size =
      (Target->getGotHeaderEntriesNum() + Entries.size()) * sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  Target->writeGotHeaderEntries(Buf);
  Buf += Target->getGotHeaderEntriesNum() * sizeof(uintX_t);
  for (const SymbolBody *B : Entries) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    if (!B)
      continue;
    // MIPS has special rules to fill up GOT entries.
    // See "Global Offset Table" in Chapter 5 in the following document
    // for detailed description:
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    // As the first approach, we can just store addresses for all symbols.
    if (Config->EMachine != EM_MIPS && canBePreempted(B, false))
      continue; // The dynamic linker will take care of it.
    uintX_t VA = getSymVA<ELFT>(*B);
    write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Entry, VA);
  }
}

template <class ELFT>
PltSection<ELFT>::PltSection()
    : OutputSectionBase<ELFT>(".plt", llvm::ELF::SHT_PROGBITS,
                              llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR) {
  this->Header.sh_addralign = 16;
}

template <class ELFT> void PltSection<ELFT>::writeTo(uint8_t *Buf) {
  size_t Off = 0;
  bool LazyReloc = Target->supportsLazyRelocations();
  if (LazyReloc) {
    // First write PLT[0] entry which is special.
    Target->writePltZeroEntry(Buf, Out<ELFT>::GotPlt->getVA(), this->getVA());
    Off += Target->getPltZeroEntrySize();
  }
  for (auto &I : Entries) {
    const SymbolBody *E = I.first;
    unsigned RelOff = I.second;
    uint64_t GotVA =
        LazyReloc ? Out<ELFT>::GotPlt->getVA() : Out<ELFT>::Got->getVA();
    uint64_t GotE = LazyReloc ? Out<ELFT>::GotPlt->getEntryAddr(*E)
                              : Out<ELFT>::Got->getEntryAddr(*E);
    uint64_t Plt = this->getVA() + Off;
    Target->writePltEntry(Buf + Off, GotVA, GotE, Plt, E->PltIndex, RelOff);
    Off += Target->getPltEntrySize();
  }
}

template <class ELFT> void PltSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->PltIndex = Entries.size();
  unsigned RelOff = Target->supportsLazyRelocations()
                        ? Out<ELFT>::RelaPlt->getRelocOffset()
                        : Out<ELFT>::RelaDyn->getRelocOffset();
  Entries.push_back(std::make_pair(Sym, RelOff));
}

template <class ELFT>
typename PltSection<ELFT>::uintX_t
PltSection<ELFT>::getEntryAddr(const SymbolBody &B) const {
  return this->getVA() + Target->getPltZeroEntrySize() +
         B.PltIndex * Target->getPltEntrySize();
}

template <class ELFT> void PltSection<ELFT>::finalize() {
  this->Header.sh_size = Target->getPltZeroEntrySize() +
                         Entries.size() * Target->getPltEntrySize();
}

template <class ELFT>
RelocationSection<ELFT>::RelocationSection(StringRef Name, bool IsRela)
    : OutputSectionBase<ELFT>(Name,
                              IsRela ? llvm::ELF::SHT_RELA : llvm::ELF::SHT_REL,
                              llvm::ELF::SHF_ALLOC),
      IsRela(IsRela) {
  this->Header.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  this->Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

// Applies corresponding symbol and type for dynamic tls relocation.
// Returns true if relocation was handled.
template <class ELFT>
bool RelocationSection<ELFT>::applyTlsDynamicReloc(SymbolBody *Body,
                                                   uint32_t Type, Elf_Rel *P,
                                                   Elf_Rel *N) {
  if (Target->isTlsLocalDynamicReloc(Type)) {
    P->setSymbolAndType(0, Target->getTlsModuleIndexReloc(), Config->Mips64EL);
    P->r_offset = Out<ELFT>::Got->getLocalTlsIndexVA();
    return true;
  }

  if (!Body || !Target->isTlsGlobalDynamicReloc(Type))
    return false;

  if (Target->isTlsOptimized(Type, Body)) {
    P->setSymbolAndType(Body->DynamicSymbolTableIndex,
                        Target->getTlsGotReloc(), Config->Mips64EL);
    P->r_offset = Out<ELFT>::Got->getEntryAddr(*Body);
    return true;
  }

  P->setSymbolAndType(Body->DynamicSymbolTableIndex,
                      Target->getTlsModuleIndexReloc(), Config->Mips64EL);
  P->r_offset = Out<ELFT>::Got->getGlobalDynAddr(*Body);
  N->setSymbolAndType(Body->DynamicSymbolTableIndex,
                      Target->getTlsOffsetReloc(), Config->Mips64EL);
  N->r_offset = Out<ELFT>::Got->getGlobalDynAddr(*Body) + sizeof(uintX_t);
  return true;
}

template <class ELFT> void RelocationSection<ELFT>::writeTo(uint8_t *Buf) {
  const unsigned EntrySize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  for (const DynamicReloc<ELFT> &Rel : Relocs) {
    auto *P = reinterpret_cast<Elf_Rel *>(Buf);
    Buf += EntrySize;

    // Skip placeholder for global dynamic TLS relocation pair. It was already
    // handled by the previous relocation.
    if (!Rel.C || !Rel.RI)
      continue;

    InputSectionBase<ELFT> &C = *Rel.C;
    const Elf_Rel &RI = *Rel.RI;
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    const ObjectFile<ELFT> &File = *C.getFile();
    SymbolBody *Body = File.getSymbolBody(SymIndex);
    if (Body)
      Body = Body->repl();

    uint32_t Type = RI.getType(Config->Mips64EL);
    if (applyTlsDynamicReloc(Body, Type, P, reinterpret_cast<Elf_Rel *>(Buf)))
      continue;
    bool NeedsCopy = Body && Target->needsCopyRel(Type, *Body);
    bool NeedsGot = Body && Target->relocNeedsGot(Type, *Body);
    bool CanBePreempted = canBePreempted(Body, NeedsGot);
    bool LazyReloc = Body && Target->supportsLazyRelocations() &&
                     Target->relocNeedsPlt(Type, *Body);
    bool IsDynRelative = Type == Target->getRelativeReloc();

    unsigned Sym = CanBePreempted ? Body->DynamicSymbolTableIndex : 0;
    unsigned Reloc;
    if (!CanBePreempted && Body && isGnuIFunc<ELFT>(*Body))
      Reloc = Target->getIRelativeReloc();
    else if (!CanBePreempted || IsDynRelative)
      Reloc = Target->getRelativeReloc();
    else if (LazyReloc)
      Reloc = Target->getPltReloc();
    else if (NeedsGot)
      Reloc = Body->isTls() ? Target->getTlsGotReloc() : Target->getGotReloc();
    else if (NeedsCopy)
      Reloc = Target->getCopyReloc();
    else
      Reloc = Target->getDynReloc(Type);
    P->setSymbolAndType(Sym, Reloc, Config->Mips64EL);

    if (NeedsGot) {
      if (LazyReloc)
        P->r_offset = Out<ELFT>::GotPlt->getEntryAddr(*Body);
      else
        P->r_offset = Out<ELFT>::Got->getEntryAddr(*Body);
    } else if (NeedsCopy) {
      P->r_offset = Out<ELFT>::Bss->getVA() +
                    dyn_cast<SharedSymbol<ELFT>>(Body)->OffsetInBSS;
    } else {
      P->r_offset = C.getOffset(RI.r_offset) + C.OutSec->getVA();
    }

    uintX_t OrigAddend = 0;
    if (IsRela && !NeedsGot)
      OrigAddend = static_cast<const Elf_Rela &>(RI).r_addend;

    uintX_t Addend;
    if (NeedsCopy)
      Addend = 0;
    else if (CanBePreempted || IsDynRelative)
      Addend = OrigAddend;
    else if (Body)
      Addend = getSymVA<ELFT>(*Body) + OrigAddend;
    else if (IsRela)
      Addend =
          getLocalRelTarget(File, static_cast<const Elf_Rela &>(RI),
                            getAddend<ELFT>(static_cast<const Elf_Rela &>(RI)));
    else
      Addend = getLocalRelTarget(File, RI, 0);

    if (IsRela)
      static_cast<Elf_Rela *>(P)->r_addend = Addend;
  }
}

template <class ELFT> unsigned RelocationSection<ELFT>::getRelocOffset() {
  const unsigned EntrySize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  return EntrySize * Relocs.size();
}

template <class ELFT> void RelocationSection<ELFT>::finalize() {
  this->Header.sh_link = Static ? Out<ELFT>::SymTab->SectionIndex
                                : Out<ELFT>::DynSymTab->SectionIndex;
  this->Header.sh_size = Relocs.size() * this->Header.sh_entsize;
}

template <class ELFT>
InterpSection<ELFT>::InterpSection()
    : OutputSectionBase<ELFT>(".interp", llvm::ELF::SHT_PROGBITS,
                              llvm::ELF::SHF_ALLOC) {
  this->Header.sh_size = Config->DynamicLinker.size() + 1;
  this->Header.sh_addralign = 1;
}

template <class ELFT>
void OutputSectionBase<ELFT>::writeHeaderTo(Elf_Shdr *SHdr) {
  Header.sh_name = Out<ELFT>::ShStrTab->getOffset(Name);
  *SHdr = Header;
}

template <class ELFT> void InterpSection<ELFT>::writeTo(uint8_t *Buf) {
  memcpy(Buf, Config->DynamicLinker.data(), Config->DynamicLinker.size());
}

template <class ELFT>
HashTableSection<ELFT>::HashTableSection()
    : OutputSectionBase<ELFT>(".hash", llvm::ELF::SHT_HASH,
                              llvm::ELF::SHF_ALLOC) {
  this->Header.sh_entsize = sizeof(Elf_Word);
  this->Header.sh_addralign = sizeof(Elf_Word);
}

static uint32_t hashSysv(StringRef Name) {
  uint32_t H = 0;
  for (char C : Name) {
    H = (H << 4) + C;
    uint32_t G = H & 0xf0000000;
    if (G)
      H ^= G >> 24;
    H &= ~G;
  }
  return H;
}

template <class ELFT> void HashTableSection<ELFT>::finalize() {
  this->Header.sh_link = Out<ELFT>::DynSymTab->SectionIndex;

  unsigned NumEntries = 2;                 // nbucket and nchain.
  NumEntries += Out<ELFT>::DynSymTab->getNumSymbols(); // The chain entries.

  // Create as many buckets as there are symbols.
  // FIXME: This is simplistic. We can try to optimize it, but implementing
  // support for SHT_GNU_HASH is probably even more profitable.
  NumEntries += Out<ELFT>::DynSymTab->getNumSymbols();
  this->Header.sh_size = NumEntries * sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  unsigned NumSymbols = Out<ELFT>::DynSymTab->getNumSymbols();
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NumSymbols; // nbucket
  *P++ = NumSymbols; // nchain

  Elf_Word *Buckets = P;
  Elf_Word *Chains = P + NumSymbols;

  for (SymbolBody *Body : Out<ELFT>::DynSymTab->getSymbols()) {
    StringRef Name = Body->getName();
    unsigned I = Body->DynamicSymbolTableIndex;
    uint32_t Hash = hashSysv(Name) % NumSymbols;
    Chains[I] = Buckets[Hash];
    Buckets[Hash] = I;
  }
}

static uint32_t hashGnu(StringRef Name) {
  uint32_t H = 5381;
  for (uint8_t C : Name)
    H = (H << 5) + H + C;
  return H;
}

template <class ELFT>
GnuHashTableSection<ELFT>::GnuHashTableSection()
    : OutputSectionBase<ELFT>(".gnu.hash", llvm::ELF::SHT_GNU_HASH,
                              llvm::ELF::SHF_ALLOC) {
  this->Header.sh_entsize = ELFT::Is64Bits ? 0 : 4;
  this->Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

template <class ELFT>
unsigned GnuHashTableSection<ELFT>::calcNBuckets(unsigned NumHashed) {
  if (!NumHashed)
    return 0;

  // These values are prime numbers which are not greater than 2^(N-1) + 1.
  // In result, for any particular NumHashed we return a prime number
  // which is not greater than NumHashed.
  static const unsigned Primes[] = {
      1,   1,    3,    3,    7,    13,    31,    61,    127,   251,
      509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071};

  return Primes[std::min<unsigned>(Log2_32_Ceil(NumHashed),
                                   array_lengthof(Primes) - 1)];
}

// Bloom filter estimation: at least 8 bits for each hashed symbol.
// GNU Hash table requirement: it should be a power of 2,
//   the minimum value is 1, even for an empty table.
// Expected results for a 32-bit target:
//   calcMaskWords(0..4)   = 1
//   calcMaskWords(5..8)   = 2
//   calcMaskWords(9..16)  = 4
// For a 64-bit target:
//   calcMaskWords(0..8)   = 1
//   calcMaskWords(9..16)  = 2
//   calcMaskWords(17..32) = 4
template <class ELFT>
unsigned GnuHashTableSection<ELFT>::calcMaskWords(unsigned NumHashed) {
  if (!NumHashed)
    return 1;
  return NextPowerOf2((NumHashed - 1) / sizeof(Elf_Off));
}

template <class ELFT> void GnuHashTableSection<ELFT>::finalize() {
  unsigned NumHashed = HashedSymbols.size();
  NBuckets = calcNBuckets(NumHashed);
  MaskWords = calcMaskWords(NumHashed);
  // Second hash shift estimation: just predefined values.
  Shift2 = ELFT::Is64Bits ? 6 : 5;

  this->Header.sh_link = Out<ELFT>::DynSymTab->SectionIndex;
  this->Header.sh_size = sizeof(Elf_Word) * 4            // Header
                         + sizeof(Elf_Off) * MaskWords   // Bloom Filter
                         + sizeof(Elf_Word) * NBuckets   // Hash Buckets
                         + sizeof(Elf_Word) * NumHashed; // Hash Values
}

template <class ELFT> void GnuHashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  writeHeader(Buf);
  if (HashedSymbols.empty())
    return;
  writeBloomFilter(Buf);
  writeHashTable(Buf);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeHeader(uint8_t *&Buf) {
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NBuckets;
  *P++ = Out<ELFT>::DynSymTab->getNumSymbols() - HashedSymbols.size();
  *P++ = MaskWords;
  *P++ = Shift2;
  Buf = reinterpret_cast<uint8_t *>(P);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeBloomFilter(uint8_t *&Buf) {
  unsigned C = sizeof(Elf_Off) * 8;

  auto *Masks = reinterpret_cast<Elf_Off *>(Buf);
  for (const HashedSymbolData &Item : HashedSymbols) {
    size_t Pos = (Item.Hash / C) & (MaskWords - 1);
    uintX_t V = (uintX_t(1) << (Item.Hash % C)) |
                (uintX_t(1) << ((Item.Hash >> Shift2) % C));
    Masks[Pos] |= V;
  }
  Buf += sizeof(Elf_Off) * MaskWords;
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeHashTable(uint8_t *Buf) {
  Elf_Word *Buckets = reinterpret_cast<Elf_Word *>(Buf);
  Elf_Word *Values = Buckets + NBuckets;

  int PrevBucket = -1;
  int I = 0;
  for (const HashedSymbolData &Item : HashedSymbols) {
    int Bucket = Item.Hash % NBuckets;
    assert(PrevBucket <= Bucket);
    if (Bucket != PrevBucket) {
      Buckets[Bucket] = Item.Body->DynamicSymbolTableIndex;
      PrevBucket = Bucket;
      if (I > 0)
        Values[I - 1] |= 1;
    }
    Values[I] = Item.Hash & ~1;
    ++I;
  }
  if (I > 0)
    Values[I - 1] |= 1;
}

static bool includeInGnuHashTable(SymbolBody *B) {
  // Assume that includeInDynamicSymtab() is already checked.
  return !B->isUndefined();
}

template <class ELFT>
void GnuHashTableSection<ELFT>::addSymbols(std::vector<SymbolBody *> &Symbols) {
  std::vector<SymbolBody *> NotHashed;
  NotHashed.reserve(Symbols.size());
  HashedSymbols.reserve(Symbols.size());
  for (SymbolBody *B : Symbols) {
    if (includeInGnuHashTable(B))
      HashedSymbols.push_back(HashedSymbolData{B, hashGnu(B->getName())});
    else
      NotHashed.push_back(B);
  }
  if (HashedSymbols.empty())
    return;

  unsigned NBuckets = calcNBuckets(HashedSymbols.size());
  std::stable_sort(HashedSymbols.begin(), HashedSymbols.end(),
                   [&](const HashedSymbolData &L, const HashedSymbolData &R) {
                     return L.Hash % NBuckets < R.Hash % NBuckets;
                   });

  Symbols = std::move(NotHashed);
  for (const HashedSymbolData &Item : HashedSymbols)
    Symbols.push_back(Item.Body);
}

template <class ELFT>
DynamicSection<ELFT>::DynamicSection(SymbolTable<ELFT> &SymTab)
    : OutputSectionBase<ELFT>(".dynamic", llvm::ELF::SHT_DYNAMIC,
                              llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE),
      SymTab(SymTab) {
  Elf_Shdr &Header = this->Header;
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;

  // .dynamic section is not writable on MIPS.
  // See "Special Section" in Chapter 4 in the following document:
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (Config->EMachine == EM_MIPS)
    Header.sh_flags = llvm::ELF::SHF_ALLOC;
}

template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Header.sh_size)
    return; // Already finalized.

  Elf_Shdr &Header = this->Header;
  Header.sh_link = Out<ELFT>::DynStrTab->SectionIndex;

  unsigned NumEntries = 0;
  if (Out<ELFT>::RelaDyn->hasRelocs()) {
    ++NumEntries; // DT_RELA / DT_REL
    ++NumEntries; // DT_RELASZ / DT_RELSZ
    ++NumEntries; // DT_RELAENT / DT_RELENT
  }
  if (Out<ELFT>::RelaPlt && Out<ELFT>::RelaPlt->hasRelocs()) {
    ++NumEntries; // DT_JMPREL
    ++NumEntries; // DT_PLTRELSZ
    ++NumEntries; // DT_PLTGOT / DT_MIPS_PLTGOT
    ++NumEntries; // DT_PLTREL
  }

  ++NumEntries; // DT_SYMTAB
  ++NumEntries; // DT_SYMENT
  ++NumEntries; // DT_STRTAB
  ++NumEntries; // DT_STRSZ
  if (Out<ELFT>::GnuHashTab)
    ++NumEntries; // DT_GNU_HASH
  if (Out<ELFT>::HashTab)
    ++NumEntries; // DT_HASH

  if (!Config->RPath.empty()) {
    ++NumEntries; // DT_RUNPATH / DT_RPATH
    Out<ELFT>::DynStrTab->add(Config->RPath);
  }

  if (!Config->SoName.empty()) {
    ++NumEntries; // DT_SONAME
    Out<ELFT>::DynStrTab->add(Config->SoName);
  }

  if (PreInitArraySec)
    NumEntries += 2;
  if (InitArraySec)
    NumEntries += 2;
  if (FiniArraySec)
    NumEntries += 2;

  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles()) {
    if (!F->isNeeded())
      continue;
    Out<ELFT>::DynStrTab->add(F->getSoName());
    ++NumEntries;
  }

  if (Symbol *S = SymTab.getSymbols().lookup(Config->Init))
    InitSym = S->Body;
  if (Symbol *S = SymTab.getSymbols().lookup(Config->Fini))
    FiniSym = S->Body;
  if (InitSym)
    ++NumEntries; // DT_INIT
  if (FiniSym)
    ++NumEntries; // DT_FINI

  if (Config->Bsymbolic)
    DtFlags |= DF_SYMBOLIC;
  if (Config->ZNodelete)
    DtFlags1 |= DF_1_NODELETE;
  if (Config->ZNow) {
    DtFlags |= DF_BIND_NOW;
    DtFlags1 |= DF_1_NOW;
  }
  if (Config->ZOrigin) {
    DtFlags |= DF_ORIGIN;
    DtFlags1 |= DF_1_ORIGIN;
  }

  if (DtFlags)
    ++NumEntries; // DT_FLAGS
  if (DtFlags1)
    ++NumEntries; // DT_FLAGS_1

  if (Config->EMachine == EM_MIPS) {
    ++NumEntries; // DT_MIPS_RLD_VERSION
    ++NumEntries; // DT_MIPS_FLAGS
    ++NumEntries; // DT_MIPS_BASE_ADDRESS
    ++NumEntries; // DT_MIPS_SYMTABNO
    ++NumEntries; // DT_MIPS_LOCAL_GOTNO
    ++NumEntries; // DT_MIPS_GOTSYM;
    ++NumEntries; // DT_PLTGOT
    if (Out<ELFT>::MipsRldMap)
      ++NumEntries; // DT_MIPS_RLD_MAP
  }

  ++NumEntries; // DT_NULL

  Header.sh_size = NumEntries * Header.sh_entsize;
}

template <class ELFT> void DynamicSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

  auto WritePtr = [&](int32_t Tag, uint64_t Val) {
    P->d_tag = Tag;
    P->d_un.d_ptr = Val;
    ++P;
  };

  auto WriteVal = [&](int32_t Tag, uint32_t Val) {
    P->d_tag = Tag;
    P->d_un.d_val = Val;
    ++P;
  };

  if (Out<ELFT>::RelaDyn->hasRelocs()) {
    bool IsRela = Out<ELFT>::RelaDyn->isRela();
    WritePtr(IsRela ? DT_RELA : DT_REL, Out<ELFT>::RelaDyn->getVA());
    WriteVal(IsRela ? DT_RELASZ : DT_RELSZ, Out<ELFT>::RelaDyn->getSize());
    WriteVal(IsRela ? DT_RELAENT : DT_RELENT,
             IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel));
  }
  if (Out<ELFT>::RelaPlt && Out<ELFT>::RelaPlt->hasRelocs()) {
    WritePtr(DT_JMPREL, Out<ELFT>::RelaPlt->getVA());
    WriteVal(DT_PLTRELSZ, Out<ELFT>::RelaPlt->getSize());
    // On MIPS, the address of the .got.plt section is stored in
    // the DT_MIPS_PLTGOT entry because the DT_PLTGOT entry points to
    // the .got section. See "Dynamic Section" in the following document:
    // https://sourceware.org/ml/binutils/2008-07/txt00000.txt
    WritePtr((Config->EMachine == EM_MIPS) ? DT_MIPS_PLTGOT : DT_PLTGOT,
             Out<ELFT>::GotPlt->getVA());
    WriteVal(DT_PLTREL, Out<ELFT>::RelaPlt->isRela() ? DT_RELA : DT_REL);
  }

  WritePtr(DT_SYMTAB, Out<ELFT>::DynSymTab->getVA());
  WritePtr(DT_SYMENT, sizeof(Elf_Sym));
  WritePtr(DT_STRTAB, Out<ELFT>::DynStrTab->getVA());
  WriteVal(DT_STRSZ, Out<ELFT>::DynStrTab->data().size());
  if (Out<ELFT>::GnuHashTab)
    WritePtr(DT_GNU_HASH, Out<ELFT>::GnuHashTab->getVA());
  if (Out<ELFT>::HashTab)
    WritePtr(DT_HASH, Out<ELFT>::HashTab->getVA());

  // If --enable-new-dtags is set, lld emits DT_RUNPATH
  // instead of DT_RPATH. The two tags are functionally
  // equivalent except for the following:
  // - DT_RUNPATH is searched after LD_LIBRARY_PATH, while
  //   DT_RPATH is searched before.
  // - DT_RUNPATH is used only to search for direct
  //   dependencies of the object it's contained in, while
  //   DT_RPATH is used for indirect dependencies as well.
  if (!Config->RPath.empty())
    WriteVal(Config->EnableNewDtags ? DT_RUNPATH : DT_RPATH,
             Out<ELFT>::DynStrTab->getOffset(Config->RPath));

  if (!Config->SoName.empty())
    WriteVal(DT_SONAME, Out<ELFT>::DynStrTab->getOffset(Config->SoName));

  auto WriteArray = [&](int32_t T1, int32_t T2,
                        const OutputSectionBase<ELFT> *Sec) {
    if (!Sec)
      return;
    WritePtr(T1, Sec->getVA());
    WriteVal(T2, Sec->getSize());
  };
  WriteArray(DT_PREINIT_ARRAY, DT_PREINIT_ARRAYSZ, PreInitArraySec);
  WriteArray(DT_INIT_ARRAY, DT_INIT_ARRAYSZ, InitArraySec);
  WriteArray(DT_FINI_ARRAY, DT_FINI_ARRAYSZ, FiniArraySec);

  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles())
    if (F->isNeeded())
      WriteVal(DT_NEEDED, Out<ELFT>::DynStrTab->getOffset(F->getSoName()));

  if (InitSym)
    WritePtr(DT_INIT, getSymVA<ELFT>(*InitSym));
  if (FiniSym)
    WritePtr(DT_FINI, getSymVA<ELFT>(*FiniSym));
  if (DtFlags)
    WriteVal(DT_FLAGS, DtFlags);
  if (DtFlags1)
    WriteVal(DT_FLAGS_1, DtFlags1);

  // See "Dynamic Section" in Chapter 5 in the following document
  // for detailed description:
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (Config->EMachine == EM_MIPS) {
    WriteVal(DT_MIPS_RLD_VERSION, 1);
    WriteVal(DT_MIPS_FLAGS, RHF_NOTPOT);
    WritePtr(DT_MIPS_BASE_ADDRESS, Target->getVAStart());
    WriteVal(DT_MIPS_SYMTABNO, Out<ELFT>::DynSymTab->getNumSymbols());
    WriteVal(DT_MIPS_LOCAL_GOTNO, Out<ELFT>::Got->getMipsLocalEntriesNum());
    if (const SymbolBody *B = Out<ELFT>::Got->getMipsFirstGlobalEntry())
      WriteVal(DT_MIPS_GOTSYM, B->DynamicSymbolTableIndex);
    else
      WriteVal(DT_MIPS_GOTSYM, Out<ELFT>::DynSymTab->getNumSymbols());
    WritePtr(DT_PLTGOT, Out<ELFT>::Got->getVA());
    if (Out<ELFT>::MipsRldMap)
      WritePtr(DT_MIPS_RLD_MAP, Out<ELFT>::MipsRldMap->getVA());
  }

  WriteVal(DT_NULL, 0);
}

template <class ELFT>
OutputSection<ELFT>::OutputSection(StringRef Name, uint32_t sh_type,
                                   uintX_t sh_flags)
    : OutputSectionBase<ELFT>(Name, sh_type, sh_flags) {}

template <class ELFT>
void OutputSection<ELFT>::addSection(InputSection<ELFT> *C) {
  Sections.push_back(C);
  C->OutSec = this;
  uint32_t Align = C->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  uintX_t Off = this->Header.sh_size;
  Off = RoundUpToAlignment(Off, Align);
  C->OutSecOff = Off;
  Off += C->getSize();
  this->Header.sh_size = Off;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t lld::elf2::getSymVA(const SymbolBody &S) {
  switch (S.kind()) {
  case SymbolBody::DefinedSyntheticKind: {
    auto &D = cast<DefinedSynthetic<ELFT>>(S);
    return D.Section.getVA() + D.Sym.st_value;
  }
  case SymbolBody::DefinedAbsoluteKind:
    return cast<DefinedAbsolute<ELFT>>(S).Sym.st_value;
  case SymbolBody::DefinedRegularKind: {
    const auto &DR = cast<DefinedRegular<ELFT>>(S);
    InputSectionBase<ELFT> &SC = DR.Section;
    if (DR.Sym.getType() == STT_TLS)
      return SC.OutSec->getVA() + SC.getOffset(DR.Sym) -
             Out<ELFT>::TlsPhdr->p_vaddr;
    return SC.OutSec->getVA() + SC.getOffset(DR.Sym);
  }
  case SymbolBody::DefinedCommonKind:
    return Out<ELFT>::Bss->getVA() + cast<DefinedCommon<ELFT>>(S).OffsetInBSS;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(S);
    if (SS.NeedsCopy)
      return Out<ELFT>::Bss->getVA() + SS.OffsetInBSS;
    return 0;
  }
  case SymbolBody::UndefinedElfKind:
  case SymbolBody::UndefinedKind:
    return 0;
  case SymbolBody::LazyKind:
    assert(S.isUsedInRegularObj() && "Lazy symbol reached writer");
    return 0;
  }
  llvm_unreachable("Invalid symbol kind");
}

// Returns a VA which a relocatin RI refers to. Used only for local symbols.
// For non-local symbols, use getSymVA instead.
template <class ELFT, bool IsRela>
typename ELFFile<ELFT>::uintX_t
lld::elf2::getLocalRelTarget(const ObjectFile<ELFT> &File,
                             const Elf_Rel_Impl<ELFT, IsRela> &RI,
                             typename ELFFile<ELFT>::uintX_t Addend) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;

  // PPC64 has a special relocation representing the TOC base pointer
  // that does not have a corresponding symbol.
  if (Config->EMachine == EM_PPC64 && RI.getType(false) == R_PPC64_TOC)
    return getPPC64TocBase() + Addend;

  const Elf_Sym *Sym =
      File.getObj().getRelocationSymbol(&RI, File.getSymbolTable());

  if (!Sym)
    error("Unsupported relocation without symbol");

  InputSectionBase<ELFT> *Section = File.getSection(*Sym);

  if (Sym->getType() == STT_TLS)
    return (Section->OutSec->getVA() + Section->getOffset(*Sym) + Addend) -
           Out<ELFT>::TlsPhdr->p_vaddr;

  // According to the ELF spec reference to a local symbol from outside
  // the group are not allowed. Unfortunately .eh_frame breaks that rule
  // and must be treated specially. For now we just replace the symbol with
  // 0.
  if (Section == &InputSection<ELFT>::Discarded || !Section->isLive())
    return Addend;

  uintX_t VA = Section->OutSec->getVA();
  if (isa<InputSection<ELFT>>(Section))
    return VA + Section->getOffset(*Sym) + Addend;

  uintX_t Offset = Sym->st_value;
  if (Sym->getType() == STT_SECTION) {
    Offset += Addend;
    Addend = 0;
  }
  return VA + cast<MergeInputSection<ELFT>>(Section)->getOffset(Offset) +
         Addend;
}

// Returns true if a symbol can be replaced at load-time by a symbol
// with the same name defined in other ELF executable or DSO.
bool lld::elf2::canBePreempted(const SymbolBody *Body, bool NeedsGot) {
  if (!Body)
    return false;  // Body is a local symbol.
  if (Body->isShared())
    return true;

  if (Body->isUndefined()) {
    if (!Body->isWeak())
      return true;

    // This is an horrible corner case. Ideally we would like to say that any
    // undefined symbol can be preempted so that the dynamic linker has a
    // chance of finding it at runtime.
    //
    // The problem is that the code sequence used to test for weak undef
    // functions looks like
    // if (func) func()
    // If the code is -fPIC the first reference is a load from the got and
    // everything works.
    // If the code is not -fPIC there is no reasonable way to solve it:
    // * A relocation writing to the text segment will fail (it is ro).
    // * A copy relocation doesn't work for functions.
    // * The trick of using a plt entry as the address would fail here since
    //   the plt entry would have a non zero address.
    // Since we cannot do anything better, we just resolve the symbol to 0 and
    // don't produce a dynamic relocation.
    //
    // As an extra hack, assume that if we are producing a shared library the
    // user knows what he or she is doing and can handle a dynamic relocation.
    return Config->Shared || NeedsGot;
  }
  if (!Config->Shared)
    return false;
  return Body->getVisibility() == STV_DEFAULT;
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  for (InputSection<ELFT> *C : Sections)
    C->writeTo(Buf);
}

template <class ELFT>
EHOutputSection<ELFT>::EHOutputSection(StringRef Name, uint32_t sh_type,
                                       uintX_t sh_flags)
    : OutputSectionBase<ELFT>(Name, sh_type, sh_flags) {}

template <class ELFT>
EHRegion<ELFT>::EHRegion(EHInputSection<ELFT> *S, unsigned Index)
    : S(S), Index(Index) {}

template <class ELFT> StringRef EHRegion<ELFT>::data() const {
  ArrayRef<uint8_t> SecData = S->getSectionData();
  ArrayRef<std::pair<uintX_t, uintX_t>> Offsets = S->Offsets;
  size_t Start = Offsets[Index].first;
  size_t End =
      Index == Offsets.size() - 1 ? SecData.size() : Offsets[Index + 1].first;
  return StringRef((const char *)SecData.data() + Start, End - Start);
}

template <class ELFT>
Cie<ELFT>::Cie(EHInputSection<ELFT> *S, unsigned Index)
    : EHRegion<ELFT>(S, Index) {}

template <class ELFT>
template <bool IsRela>
void EHOutputSection<ELFT>::addSectionAux(
    EHInputSection<ELFT> *S,
    iterator_range<const Elf_Rel_Impl<ELFT, IsRela> *> Rels) {
  const endianness E = ELFT::TargetEndianness;

  S->OutSec = this;
  uint32_t Align = S->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  Sections.push_back(S);

  ArrayRef<uint8_t> SecData = S->getSectionData();
  ArrayRef<uint8_t> D = SecData;
  uintX_t Offset = 0;
  auto RelI = Rels.begin();
  auto RelE = Rels.end();

  DenseMap<unsigned, unsigned> OffsetToIndex;
  while (!D.empty()) {
    unsigned Index = S->Offsets.size();
    S->Offsets.push_back(std::make_pair(Offset, -1));

    uintX_t Length = readEntryLength(D);
    StringRef Entry((const char *)D.data(), Length);

    while (RelI != RelE && RelI->r_offset < Offset)
      ++RelI;
    uintX_t NextOffset = Offset + Length;
    bool HasReloc = RelI != RelE && RelI->r_offset < NextOffset;

    uint32_t ID = read32<E>(D.data() + 4);
    if (ID == 0) {
      // CIE
      Cie<ELFT> C(S, Index);

      StringRef Personality;
      if (HasReloc) {
        uint32_t SymIndex = RelI->getSymbol(Config->Mips64EL);
        SymbolBody &Body = *S->getFile()->getSymbolBody(SymIndex)->repl();
        Personality = Body.getName();
      }

      std::pair<StringRef, StringRef> CieInfo(Entry, Personality);
      auto P = CieMap.insert(std::make_pair(CieInfo, Cies.size()));
      if (P.second) {
        Cies.push_back(C);
        this->Header.sh_size += RoundUpToAlignment(Length, sizeof(uintX_t));
      }
      OffsetToIndex[Offset] = P.first->second;
    } else {
      if (!HasReloc)
        error("FDE doesn't reference another section");
      InputSectionBase<ELFT> *Target = S->getRelocTarget(*RelI);
      if (Target != &InputSection<ELFT>::Discarded && Target->isLive()) {
        uint32_t CieOffset = Offset + 4 - ID;
        auto I = OffsetToIndex.find(CieOffset);
        if (I == OffsetToIndex.end())
          error("Invalid CIE reference");
        Cies[I->second].Fdes.push_back(EHRegion<ELFT>(S, Index));
        this->Header.sh_size += RoundUpToAlignment(Length, sizeof(uintX_t));
      }
    }

    Offset = NextOffset;
    D = D.slice(Length);
  }
}

template <class ELFT>
typename EHOutputSection<ELFT>::uintX_t
EHOutputSection<ELFT>::readEntryLength(ArrayRef<uint8_t> D) {
  const endianness E = ELFT::TargetEndianness;

  if (D.size() < 4)
    error("Truncated CIE/FDE length");
  uint64_t Len = read32<E>(D.data());
  if (Len < UINT32_MAX) {
    if (Len > (UINT32_MAX - 4))
      error("CIE/FIE size is too large");
    if (Len + 4 > D.size())
      error("CIE/FIE ends past the end of the section");
    return Len + 4;
  }

  if (D.size() < 12)
    error("Truncated CIE/FDE length");
  Len = read64<E>(D.data() + 4);
  if (Len > (UINT64_MAX - 12))
    error("CIE/FIE size is too large");
  if (Len + 12 > D.size())
    error("CIE/FIE ends past the end of the section");
  return Len + 12;
}

template <class ELFT>
void EHOutputSection<ELFT>::addSection(EHInputSection<ELFT> *S) {
  const Elf_Shdr *RelSec = S->RelocSection;
  if (!RelSec)
    return addSectionAux(
        S, make_range((const Elf_Rela *)nullptr, (const Elf_Rela *)nullptr));
  ELFFile<ELFT> &Obj = S->getFile()->getObj();
  if (RelSec->sh_type == SHT_RELA)
    return addSectionAux(S, Obj.relas(RelSec));
  return addSectionAux(S, Obj.rels(RelSec));
}

template <class ELFT> void EHOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;
  size_t Offset = 0;
  for (const Cie<ELFT> &C : Cies) {
    size_t CieOffset = Offset;

    StringRef CieData = C.data();
    memcpy(Buf + Offset, CieData.data(), CieData.size());
    C.S->Offsets[C.Index].second = Offset;
    Offset += RoundUpToAlignment(CieData.size(), sizeof(uintX_t));

    for (const EHRegion<ELFT> &F : C.Fdes) {
      StringRef FdeData = F.data();
      uintX_t Len = RoundUpToAlignment(FdeData.size(), sizeof(uintX_t));
      write32<E>(Buf + Offset, Len - 4);                    // Length
      write32<E>(Buf + Offset + 4, Offset + 4 - CieOffset); // Pointer
      memcpy(Buf + Offset + 8, FdeData.data() + 8, FdeData.size() - 8);
      F.S->Offsets[F.Index].second = Offset;
      Offset += Len;
    }
  }

  for (EHInputSection<ELFT> *S : Sections) {
    const Elf_Shdr *RelSec = S->RelocSection;
    if (!RelSec)
      continue;
    ELFFile<ELFT> &EObj = S->getFile()->getObj();
    if (RelSec->sh_type == SHT_RELA)
      S->relocate(Buf, nullptr, EObj.relas(RelSec));
    else
      S->relocate(Buf, nullptr, EObj.rels(RelSec));
  }
}

template <class ELFT>
MergeOutputSection<ELFT>::MergeOutputSection(StringRef Name, uint32_t sh_type,
                                             uintX_t sh_flags)
    : OutputSectionBase<ELFT>(Name, sh_type, sh_flags) {}

template <class ELFT> void MergeOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (shouldTailMerge()) {
    StringRef Data = Builder.data();
    memcpy(Buf, Data.data(), Data.size());
    return;
  }
  for (const std::pair<StringRef, size_t> &P : Builder.getMap()) {
    StringRef Data = P.first;
    memcpy(Buf + P.second, Data.data(), Data.size());
  }
}

static size_t findNull(StringRef S, size_t EntSize) {
  // Optimize the common case.
  if (EntSize == 1)
    return S.find(0);

  for (unsigned I = 0, N = S.size(); I != N; I += EntSize) {
    const char *B = S.begin() + I;
    if (std::all_of(B, B + EntSize, [](char C) { return C == 0; }))
      return I;
  }
  return StringRef::npos;
}

template <class ELFT>
void MergeOutputSection<ELFT>::addSection(MergeInputSection<ELFT> *S) {
  S->OutSec = this;
  uint32_t Align = S->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  ArrayRef<uint8_t> D = S->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  uintX_t EntSize = S->getSectionHdr()->sh_entsize;
  uintX_t Offset = 0;

  if (this->Header.sh_flags & SHF_STRINGS) {
    while (!Data.empty()) {
      size_t End = findNull(Data, EntSize);
      if (End == StringRef::npos)
        error("String is not null terminated");
      StringRef Entry = Data.substr(0, End + EntSize);
      uintX_t OutputOffset = Builder.add(Entry);
      if (shouldTailMerge())
        OutputOffset = -1;
      S->Offsets.push_back(std::make_pair(Offset, OutputOffset));
      uintX_t Size = End + EntSize;
      Data = Data.substr(Size);
      Offset += Size;
    }
  } else {
    for (unsigned I = 0, N = Data.size(); I != N; I += EntSize) {
      StringRef Entry = Data.substr(I, EntSize);
      size_t OutputOffset = Builder.add(Entry);
      S->Offsets.push_back(std::make_pair(Offset, OutputOffset));
      Offset += EntSize;
    }
  }
}

template <class ELFT>
unsigned MergeOutputSection<ELFT>::getOffset(StringRef Val) {
  return Builder.getOffset(Val);
}

template <class ELFT> bool MergeOutputSection<ELFT>::shouldTailMerge() const {
  return Config->Optimize >= 2 && this->Header.sh_flags & SHF_STRINGS;
}

template <class ELFT> void MergeOutputSection<ELFT>::finalize() {
  if (shouldTailMerge())
    Builder.finalize();
  this->Header.sh_size = Builder.getSize();
}

template <class ELFT>
StringTableSection<ELFT>::StringTableSection(StringRef Name, bool Dynamic)
    : OutputSectionBase<ELFT>(Name, llvm::ELF::SHT_STRTAB,
                              Dynamic ? (uintX_t)llvm::ELF::SHF_ALLOC : 0),
      Dynamic(Dynamic) {
  this->Header.sh_addralign = 1;
}

template <class ELFT> void StringTableSection<ELFT>::writeTo(uint8_t *Buf) {
  StringRef Data = StrTabBuilder.data();
  memcpy(Buf, Data.data(), Data.size());
}

template <class ELFT> bool lld::elf2::includeInSymtab(const SymbolBody &B) {
  if (!B.isUsedInRegularObj())
    return false;

  // Don't include synthetic symbols like __init_array_start in every output.
  if (auto *U = dyn_cast<DefinedAbsolute<ELFT>>(&B))
    if (&U->Sym == &DefinedAbsolute<ELFT>::IgnoreUndef)
      return false;

  return true;
}

bool lld::elf2::includeInDynamicSymtab(const SymbolBody &B) {
  uint8_t V = B.getVisibility();
  if (V != STV_DEFAULT && V != STV_PROTECTED)
    return false;

  if (Config->ExportDynamic || Config->Shared)
    return true;
  return B.isUsedInDynamicReloc();
}

template <class ELFT>
bool lld::elf2::shouldKeepInSymtab(const ObjectFile<ELFT> &File,
                                   StringRef SymName,
                                   const typename ELFFile<ELFT>::Elf_Sym &Sym) {
  if (Sym.getType() == STT_SECTION)
    return false;

  InputSectionBase<ELFT> *Sec = File.getSection(Sym);
  // If sym references a section in a discarded group, don't keep it.
  if (Sec == &InputSection<ELFT>::Discarded)
    return false;

  if (Config->DiscardNone)
    return true;

  // In ELF assembly .L symbols are normally discarded by the assembler.
  // If the assembler fails to do so, the linker discards them if
  // * --discard-locals is used.
  // * The symbol is in a SHF_MERGE section, which is normally the reason for
  //   the assembler keeping the .L symbol.
  if (!SymName.startswith(".L") && !SymName.empty())
    return true;

  if (Config->DiscardLocals)
    return false;

  return !(Sec->getSectionHdr()->sh_flags & SHF_MERGE);
}

template <class ELFT>
SymbolTableSection<ELFT>::SymbolTableSection(
    SymbolTable<ELFT> &Table, StringTableSection<ELFT> &StrTabSec)
    : OutputSectionBase<ELFT>(
          StrTabSec.isDynamic() ? ".dynsym" : ".symtab",
          StrTabSec.isDynamic() ? llvm::ELF::SHT_DYNSYM : llvm::ELF::SHT_SYMTAB,
          StrTabSec.isDynamic() ? (uintX_t)llvm::ELF::SHF_ALLOC : 0),
      Table(Table), StrTabSec(StrTabSec) {
  typedef OutputSectionBase<ELFT> Base;
  typename Base::Elf_Shdr &Header = this->Header;

  Header.sh_entsize = sizeof(Elf_Sym);
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
}

// Orders symbols according to their positions in the GOT,
// in compliance with MIPS ABI rules.
// See "Global Offset Table" in Chapter 5 in the following document
// for detailed description:
// ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
static bool sortMipsSymbols(SymbolBody *L, SymbolBody *R) {
  if (!L->isInGot() || !R->isInGot())
    return R->isInGot();
  return L->GotIndex < R->GotIndex;
}

template <class ELFT> void SymbolTableSection<ELFT>::finalize() {
  if (this->Header.sh_size)
    return; // Already finalized.

  this->Header.sh_size = getNumSymbols() * sizeof(Elf_Sym);
  this->Header.sh_link = StrTabSec.SectionIndex;
  this->Header.sh_info = NumLocals + 1;

  if (!StrTabSec.isDynamic()) {
    std::stable_sort(Symbols.begin(), Symbols.end(),
                     [](SymbolBody *L, SymbolBody *R) {
                       return getSymbolBinding(L) == STB_LOCAL &&
                              getSymbolBinding(R) != STB_LOCAL;
                     });
    return;
  }
  if (Out<ELFT>::GnuHashTab)
    // NB: It also sorts Symbols to meet the GNU hash table requirements.
    Out<ELFT>::GnuHashTab->addSymbols(Symbols);
  else if (Config->EMachine == EM_MIPS)
    std::stable_sort(Symbols.begin(), Symbols.end(), sortMipsSymbols);
  size_t I = 0;
  for (SymbolBody *B : Symbols)
    B->DynamicSymbolTableIndex = ++I;
}

template <class ELFT>
void SymbolTableSection<ELFT>::addLocalSymbol(StringRef Name) {
  StrTabSec.add(Name);
  ++NumVisible;
  ++NumLocals;
}

template <class ELFT>
void SymbolTableSection<ELFT>::addSymbol(SymbolBody *Body) {
  StrTabSec.add(Body->getName());
  Symbols.push_back(Body);
  ++NumVisible;
}

template <class ELFT> void SymbolTableSection<ELFT>::writeTo(uint8_t *Buf) {
  Buf += sizeof(Elf_Sym);

  // All symbols with STB_LOCAL binding precede the weak and global symbols.
  // .dynsym only contains global symbols.
  if (!Config->DiscardAll && !StrTabSec.isDynamic())
    writeLocalSymbols(Buf);

  writeGlobalSymbols(Buf);
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeLocalSymbols(uint8_t *&Buf) {
  // Iterate over all input object files to copy their local symbols
  // to the output symbol table pointed by Buf.
  for (const std::unique_ptr<ObjectFile<ELFT>> &File : Table.getObjectFiles()) {
    Elf_Sym_Range Syms = File->getLocalSymbols();
    for (const Elf_Sym &Sym : Syms) {
      ErrorOr<StringRef> SymNameOrErr = Sym.getName(File->getStringTable());
      error(SymNameOrErr);
      StringRef SymName = *SymNameOrErr;
      if (!shouldKeepInSymtab<ELFT>(*File, SymName, Sym))
        continue;

      auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
      uintX_t VA = 0;
      if (Sym.st_shndx == SHN_ABS) {
        ESym->st_shndx = SHN_ABS;
        VA = Sym.st_value;
      } else {
        InputSectionBase<ELFT> *Section = File->getSection(Sym);
        if (!Section->isLive())
          continue;
        const OutputSectionBase<ELFT> *OutSec = Section->OutSec;
        ESym->st_shndx = OutSec->SectionIndex;
        VA += OutSec->getVA() + Section->getOffset(Sym);
      }
      ESym->st_name = StrTabSec.getOffset(SymName);
      ESym->st_size = Sym.st_size;
      ESym->setBindingAndType(Sym.getBinding(), Sym.getType());
      ESym->st_value = VA;
      Buf += sizeof(*ESym);
    }
  }
}

template <class ELFT>
static const typename llvm::object::ELFFile<ELFT>::Elf_Sym *
getElfSym(SymbolBody &Body) {
  if (auto *EBody = dyn_cast<Defined<ELFT>>(&Body))
    return &EBody->Sym;
  if (auto *EBody = dyn_cast<UndefinedElf<ELFT>>(&Body))
    return &EBody->Sym;
  return nullptr;
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeGlobalSymbols(uint8_t *Buf) {
  // Write the internal symbol table contents to the output symbol table
  // pointed by Buf.
  auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
  for (SymbolBody *Body : Symbols) {
    const OutputSectionBase<ELFT> *OutSec = nullptr;

    switch (Body->kind()) {
    case SymbolBody::DefinedSyntheticKind:
      OutSec = &cast<DefinedSynthetic<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedRegularKind: {
      auto *Sym = cast<DefinedRegular<ELFT>>(Body->repl());
      if (!Sym->Section.isLive())
        continue;
      OutSec = Sym->Section.OutSec;
      break;
    }
    case SymbolBody::DefinedCommonKind:
      OutSec = Out<ELFT>::Bss;
      break;
    case SymbolBody::SharedKind: {
      if (cast<SharedSymbol<ELFT>>(Body)->NeedsCopy)
        OutSec = Out<ELFT>::Bss;
      break;
    }
    case SymbolBody::UndefinedElfKind:
    case SymbolBody::UndefinedKind:
    case SymbolBody::DefinedAbsoluteKind:
    case SymbolBody::LazyKind:
      break;
    }

    StringRef Name = Body->getName();
    ESym->st_name = StrTabSec.getOffset(Name);

    unsigned char Type = STT_NOTYPE;
    uintX_t Size = 0;
    if (const Elf_Sym *InputSym = getElfSym<ELFT>(*Body)) {
      Type = InputSym->getType();
      Size = InputSym->st_size;
    }

    ESym->setBindingAndType(getSymbolBinding(Body), Type);
    ESym->st_size = Size;
    ESym->setVisibility(Body->getVisibility());
    ESym->st_value = getSymVA<ELFT>(*Body);

    if (isa<DefinedAbsolute<ELFT>>(Body))
      ESym->st_shndx = SHN_ABS;
    else if (OutSec)
      ESym->st_shndx = OutSec->SectionIndex;

    ++ESym;
  }
}

template <class ELFT>
uint8_t SymbolTableSection<ELFT>::getSymbolBinding(SymbolBody *Body) {
  uint8_t Visibility = Body->getVisibility();
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return STB_LOCAL;
  if (const Elf_Sym *ESym = getElfSym<ELFT>(*Body))
    return ESym->getBinding();
  return Body->isWeak() ? STB_WEAK : STB_GLOBAL;
}

template <class ELFT>
MipsReginfoOutputSection<ELFT>::MipsReginfoOutputSection()
    : OutputSectionBase<ELFT>(".reginfo", SHT_MIPS_REGINFO, SHF_ALLOC) {
  this->Header.sh_addralign = 4;
  this->Header.sh_entsize = sizeof(Elf_Mips_RegInfo);
  this->Header.sh_size = sizeof(Elf_Mips_RegInfo);
}

template <class ELFT>
void MipsReginfoOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *R = reinterpret_cast<Elf_Mips_RegInfo *>(Buf);
  R->ri_gp_value = getMipsGpAddr<ELFT>();
  R->ri_gprmask = GeneralMask;
}

template <class ELFT>
void MipsReginfoOutputSection<ELFT>::addSection(
    MipsReginfoInputSection<ELFT> *S) {
  GeneralMask |= S->getGeneralMask();
}

namespace lld {
namespace elf2 {
template class OutputSectionBase<ELF32LE>;
template class OutputSectionBase<ELF32BE>;
template class OutputSectionBase<ELF64LE>;
template class OutputSectionBase<ELF64BE>;

template class GotPltSection<ELF32LE>;
template class GotPltSection<ELF32BE>;
template class GotPltSection<ELF64LE>;
template class GotPltSection<ELF64BE>;

template class GotSection<ELF32LE>;
template class GotSection<ELF32BE>;
template class GotSection<ELF64LE>;
template class GotSection<ELF64BE>;

template class PltSection<ELF32LE>;
template class PltSection<ELF32BE>;
template class PltSection<ELF64LE>;
template class PltSection<ELF64BE>;

template class RelocationSection<ELF32LE>;
template class RelocationSection<ELF32BE>;
template class RelocationSection<ELF64LE>;
template class RelocationSection<ELF64BE>;

template class InterpSection<ELF32LE>;
template class InterpSection<ELF32BE>;
template class InterpSection<ELF64LE>;
template class InterpSection<ELF64BE>;

template class GnuHashTableSection<ELF32LE>;
template class GnuHashTableSection<ELF32BE>;
template class GnuHashTableSection<ELF64LE>;
template class GnuHashTableSection<ELF64BE>;

template class HashTableSection<ELF32LE>;
template class HashTableSection<ELF32BE>;
template class HashTableSection<ELF64LE>;
template class HashTableSection<ELF64BE>;

template class DynamicSection<ELF32LE>;
template class DynamicSection<ELF32BE>;
template class DynamicSection<ELF64LE>;
template class DynamicSection<ELF64BE>;

template class OutputSection<ELF32LE>;
template class OutputSection<ELF32BE>;
template class OutputSection<ELF64LE>;
template class OutputSection<ELF64BE>;

template class EHOutputSection<ELF32LE>;
template class EHOutputSection<ELF32BE>;
template class EHOutputSection<ELF64LE>;
template class EHOutputSection<ELF64BE>;

template class MipsReginfoOutputSection<ELF32LE>;
template class MipsReginfoOutputSection<ELF32BE>;
template class MipsReginfoOutputSection<ELF64LE>;
template class MipsReginfoOutputSection<ELF64BE>;

template class MergeOutputSection<ELF32LE>;
template class MergeOutputSection<ELF32BE>;
template class MergeOutputSection<ELF64LE>;
template class MergeOutputSection<ELF64BE>;

template class StringTableSection<ELF32LE>;
template class StringTableSection<ELF32BE>;
template class StringTableSection<ELF64LE>;
template class StringTableSection<ELF64BE>;

template class SymbolTableSection<ELF32LE>;
template class SymbolTableSection<ELF32BE>;
template class SymbolTableSection<ELF64LE>;
template class SymbolTableSection<ELF64BE>;

template ELFFile<ELF32LE>::uintX_t getSymVA<ELF32LE>(const SymbolBody &);
template ELFFile<ELF32BE>::uintX_t getSymVA<ELF32BE>(const SymbolBody &);
template ELFFile<ELF64LE>::uintX_t getSymVA<ELF64LE>(const SymbolBody &);
template ELFFile<ELF64BE>::uintX_t getSymVA<ELF64BE>(const SymbolBody &);

template ELFFile<ELF32LE>::uintX_t
getLocalRelTarget(const ObjectFile<ELF32LE> &,
                  const ELFFile<ELF32LE>::Elf_Rel &,
                  ELFFile<ELF32LE>::uintX_t Addend);
template ELFFile<ELF32BE>::uintX_t
getLocalRelTarget(const ObjectFile<ELF32BE> &,
                  const ELFFile<ELF32BE>::Elf_Rel &,
                  ELFFile<ELF32BE>::uintX_t Addend);
template ELFFile<ELF64LE>::uintX_t
getLocalRelTarget(const ObjectFile<ELF64LE> &,
                  const ELFFile<ELF64LE>::Elf_Rel &,
                  ELFFile<ELF64LE>::uintX_t Addend);
template ELFFile<ELF64BE>::uintX_t
getLocalRelTarget(const ObjectFile<ELF64BE> &,
                  const ELFFile<ELF64BE>::Elf_Rel &,
                  ELFFile<ELF64BE>::uintX_t Addend);

template bool includeInSymtab<ELF32LE>(const SymbolBody &);
template bool includeInSymtab<ELF32BE>(const SymbolBody &);
template bool includeInSymtab<ELF64LE>(const SymbolBody &);
template bool includeInSymtab<ELF64BE>(const SymbolBody &);

template bool shouldKeepInSymtab<ELF32LE>(const ObjectFile<ELF32LE> &,
                                          StringRef,
                                          const ELFFile<ELF32LE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF32BE>(const ObjectFile<ELF32BE> &,
                                          StringRef,
                                          const ELFFile<ELF32BE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF64LE>(const ObjectFile<ELF64LE> &,
                                          StringRef,
                                          const ELFFile<ELF64LE>::Elf_Sym &);
template bool shouldKeepInSymtab<ELF64BE>(const ObjectFile<ELF64BE> &,
                                          StringRef,
                                          const ELFFile<ELF64BE>::Elf_Sym &);
}
}
