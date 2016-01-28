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
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/MathExtras.h"
#include <map>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
OutputSectionBase<ELFT>::OutputSectionBase(StringRef Name, uint32_t Type,
                                           uintX_t Flags)
    : Name(Name) {
  memset(&Header, 0, sizeof(Elf_Shdr));
  Header.sh_type = Type;
  Header.sh_flags = Flags;
}

template <class ELFT>
GotPltSection<ELFT>::GotPltSection()
    : OutputSectionBase<ELFT>(".got.plt", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE) {
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
    : OutputSectionBase<ELFT>(".got", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE) {
  if (Config->EMachine == EM_MIPS)
    this->Header.sh_flags |= SHF_MIPS_GPREL;
  this->Header.sh_addralign = sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody *Sym) {
  Sym->GotIndex = Entries.size();
  Entries.push_back(Sym);
}

template <class ELFT> void GotSection<ELFT>::addMipsLocalEntry() {
  ++MipsLocalEntries;
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
  return this->getVA() +
         (Target->getGotHeaderEntriesNum() + MipsLocalEntries + B.GotIndex) *
             sizeof(uintX_t);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getMipsLocalFullAddr(const SymbolBody &B) {
  return getMipsLocalEntryAddr(getSymVA<ELFT>(B));
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getMipsLocalPageAddr(uintX_t EntryValue) {
  // Initialize the entry by the %hi(EntryValue) expression
  // but without right-shifting.
  return getMipsLocalEntryAddr((EntryValue + 0x8000) & ~0xffff);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getMipsLocalEntryAddr(uintX_t EntryValue) {
  size_t NewIndex = Target->getGotHeaderEntriesNum() + MipsLocalGotPos.size();
  auto P = MipsLocalGotPos.insert(std::make_pair(EntryValue, NewIndex));
  assert(!P.second || MipsLocalGotPos.size() <= MipsLocalEntries);
  return this->getVA() + P.first->second * sizeof(uintX_t);
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
  return Target->getGotHeaderEntriesNum() + MipsLocalEntries;
}

template <class ELFT> void GotSection<ELFT>::finalize() {
  this->Header.sh_size =
      (Target->getGotHeaderEntriesNum() + MipsLocalEntries + Entries.size()) *
      sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  Target->writeGotHeaderEntries(Buf);
  for (const auto &L : MipsLocalGotPos) {
    uint8_t *Entry = Buf + L.second * sizeof(uintX_t);
    write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Entry, L.first);
  }
  Buf += Target->getGotHeaderEntriesNum() * sizeof(uintX_t);
  Buf += MipsLocalEntries * sizeof(uintX_t);
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
    : OutputSectionBase<ELFT>(".plt", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR) {
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
    : OutputSectionBase<ELFT>(Name, IsRela ? SHT_RELA : SHT_REL, SHF_ALLOC),
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
  for (const DynamicReloc<ELFT> &Rel : Relocs) {
    auto *P = reinterpret_cast<Elf_Rel *>(Buf);
    Buf += IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);

    // Skip placeholder for global dynamic TLS relocation pair. It was already
    // handled by the previous relocation.
    if (!Rel.C)
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

    // Writer::scanRelocs creates a RELATIVE reloc for some type of TLS reloc.
    // We want to write it down as is.
    if (Type == Target->getRelativeReloc()) {
      P->setSymbolAndType(0, Type, Config->Mips64EL);
      P->r_offset = C.getOffset(RI.r_offset) + C.OutSec->getVA();
      continue;
    }

    // Emit a copy relocation.
    auto *SS = dyn_cast_or_null<SharedSymbol<ELFT>>(Body);
    if (SS && SS->NeedsCopy) {
      P->setSymbolAndType(Body->DynamicSymbolTableIndex, Target->getCopyReloc(),
                          Config->Mips64EL);
      P->r_offset = Out<ELFT>::Bss->getVA() + SS->OffsetInBss;
      continue;
    }

    bool NeedsGot = Body && Target->relocNeedsGot(Type, *Body);
    bool CBP = canBePreempted(Body, NeedsGot);

    // For a symbol with STT_GNU_IFUNC type, we always create a PLT and
    // a GOT entry for the symbol, and emit an IRELATIVE reloc rather than
    // the usual JUMP_SLOT reloc for the GOT entry. For the details, you
    // want to read http://www.airs.com/blog/archives/403
    if (!CBP && Body && isGnuIFunc<ELFT>(*Body)) {
      P->setSymbolAndType(0, Target->getIRelativeReloc(), Config->Mips64EL);
      if (Out<ELFT>::GotPlt)
        P->r_offset = Out<ELFT>::GotPlt->getEntryAddr(*Body);
      else
        P->r_offset = Out<ELFT>::Got->getEntryAddr(*Body);
      continue;
    }

    bool LazyReloc = Body && Target->supportsLazyRelocations() &&
                     Target->relocNeedsPlt(Type, *Body);

    unsigned Reloc;
    if (!CBP)
      Reloc = Target->getRelativeReloc();
    else if (LazyReloc)
      Reloc = Target->getPltReloc();
    else if (NeedsGot)
      Reloc = Body->isTls() ? Target->getTlsGotReloc() : Target->getGotReloc();
    else
      Reloc = Target->getDynReloc(Type);
    P->setSymbolAndType(CBP ? Body->DynamicSymbolTableIndex : 0, Reloc,
                        Config->Mips64EL);

    if (LazyReloc)
      P->r_offset = Out<ELFT>::GotPlt->getEntryAddr(*Body);
    else if (NeedsGot)
      P->r_offset = Out<ELFT>::Got->getEntryAddr(*Body);
    else
      P->r_offset = C.getOffset(RI.r_offset) + C.OutSec->getVA();

    if (!IsRela)
      continue;

    auto R = static_cast<const Elf_Rela &>(RI);
    auto S = static_cast<Elf_Rela *>(P);
    uintX_t A = NeedsGot ? 0 : R.r_addend;
    if (CBP)
      S->r_addend = A;
    else if (Body)
      S->r_addend = getSymVA<ELFT>(*Body) + A;
    else
      S->r_addend = getLocalRelTarget(File, R, A);
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
    : OutputSectionBase<ELFT>(".interp", SHT_PROGBITS, SHF_ALLOC) {
  this->Header.sh_size = Config->DynamicLinker.size() + 1;
  this->Header.sh_addralign = 1;
}

template <class ELFT>
void OutputSectionBase<ELFT>::writeHeaderTo(Elf_Shdr *SHdr) {
  Header.sh_name = Out<ELFT>::ShStrTab->addString(Name);
  *SHdr = Header;
}

template <class ELFT> void InterpSection<ELFT>::writeTo(uint8_t *Buf) {
  memcpy(Buf, Config->DynamicLinker.data(), Config->DynamicLinker.size());
}

template <class ELFT>
HashTableSection<ELFT>::HashTableSection()
    : OutputSectionBase<ELFT>(".hash", SHT_HASH, SHF_ALLOC) {
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
    : OutputSectionBase<ELFT>(".gnu.hash", SHT_GNU_HASH, SHF_ALLOC) {
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
    : OutputSectionBase<ELFT>(".dynamic", SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE),
      SymTab(SymTab) {
  Elf_Shdr &Header = this->Header;
  Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;

  // .dynamic section is not writable on MIPS.
  // See "Special Section" in Chapter 4 in the following document:
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (Config->EMachine == EM_MIPS)
    Header.sh_flags = SHF_ALLOC;
}

template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Header.sh_size)
    return; // Already finalized.

  Elf_Shdr &Header = this->Header;
  Header.sh_link = Out<ELFT>::DynStrTab->SectionIndex;

  // Reserve strings. We know that these are the last string to be added to
  // DynStrTab and doing this here allows this function to set DT_STRSZ.
  if (!Config->RPath.empty())
    Out<ELFT>::DynStrTab->reserve(Config->RPath);
  if (!Config->SoName.empty())
    Out<ELFT>::DynStrTab->reserve(Config->SoName);
  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles())
    if (F->isNeeded())
      Out<ELFT>::DynStrTab->reserve(F->getSoName());
  Out<ELFT>::DynStrTab->finalize();

  auto Add = [=](Entry E) { Entries.push_back(E); };

  if (Out<ELFT>::RelaDyn->hasRelocs()) {
    bool IsRela = Out<ELFT>::RelaDyn->isRela();
    Add({IsRela ? DT_RELA : DT_REL, Out<ELFT>::RelaDyn});
    Add({IsRela ? DT_RELASZ : DT_RELSZ, Out<ELFT>::RelaDyn->getSize()});
    Add({IsRela ? DT_RELAENT : DT_RELENT,
         uintX_t(IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel))});
  }
  if (Out<ELFT>::RelaPlt && Out<ELFT>::RelaPlt->hasRelocs()) {
    Add({DT_JMPREL, Out<ELFT>::RelaPlt});
    Add({DT_PLTRELSZ, Out<ELFT>::RelaPlt->getSize()});
    Add({Config->EMachine == EM_MIPS ? DT_MIPS_PLTGOT : DT_PLTGOT,
         Out<ELFT>::GotPlt});
    Add({DT_PLTREL, uint64_t(Out<ELFT>::RelaPlt->isRela() ? DT_RELA : DT_REL)});
  }

  Add({DT_SYMTAB, Out<ELFT>::DynSymTab});
  Add({DT_SYMENT, sizeof(Elf_Sym)});
  Add({DT_STRTAB, Out<ELFT>::DynStrTab});
  Add({DT_STRSZ, Out<ELFT>::DynStrTab->getSize()});
  if (Out<ELFT>::GnuHashTab)
    Add({DT_GNU_HASH, Out<ELFT>::GnuHashTab});
  if (Out<ELFT>::HashTab)
    Add({DT_HASH, Out<ELFT>::HashTab});

  if (!Config->RPath.empty())
    Add({Config->EnableNewDtags ? DT_RUNPATH : DT_RPATH,
         Out<ELFT>::DynStrTab->addString(Config->RPath)});

  if (!Config->SoName.empty())
    Add({DT_SONAME, Out<ELFT>::DynStrTab->addString(Config->SoName)});

  if (PreInitArraySec) {
    Add({DT_PREINIT_ARRAY, PreInitArraySec});
    Add({DT_PREINIT_ARRAYSZ, PreInitArraySec->getSize()});
  }
  if (InitArraySec) {
    Add({DT_INIT_ARRAY, InitArraySec});
    Add({DT_INIT_ARRAYSZ, (uintX_t)InitArraySec->getSize()});
  }
  if (FiniArraySec) {
    Add({DT_FINI_ARRAY, FiniArraySec});
    Add({DT_FINI_ARRAYSZ, (uintX_t)FiniArraySec->getSize()});
  }

  for (const std::unique_ptr<SharedFile<ELFT>> &F : SymTab.getSharedFiles())
    if (F->isNeeded())
      Add({DT_NEEDED, Out<ELFT>::DynStrTab->addString(F->getSoName())});

  if (SymbolBody *B = SymTab.find(Config->Init))
    Add({DT_INIT, B});
  if (SymbolBody *B = SymTab.find(Config->Fini))
    Add({DT_FINI, B});

  uint32_t DtFlags = 0;
  uint32_t DtFlags1 = 0;
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
    Add({DT_FLAGS, DtFlags});
  if (DtFlags1)
    Add({DT_FLAGS_1, DtFlags1});

  if (!Config->Entry.empty())
    Add({DT_DEBUG, (uint64_t)0});

  if (Config->EMachine == EM_MIPS) {
    Add({DT_MIPS_RLD_VERSION, 1});
    Add({DT_MIPS_FLAGS, RHF_NOTPOT});
    Add({DT_MIPS_BASE_ADDRESS, (uintX_t)Target->getVAStart()});
    Add({DT_MIPS_SYMTABNO, Out<ELFT>::DynSymTab->getNumSymbols()});
    Add({DT_MIPS_LOCAL_GOTNO, Out<ELFT>::Got->getMipsLocalEntriesNum()});
    if (const SymbolBody *B = Out<ELFT>::Got->getMipsFirstGlobalEntry())
      Add({DT_MIPS_GOTSYM, B->DynamicSymbolTableIndex});
    else
      Add({DT_MIPS_GOTSYM, Out<ELFT>::DynSymTab->getNumSymbols()});
    Add({DT_PLTGOT, Out<ELFT>::Got});
    if (Out<ELFT>::MipsRldMap)
      Add({DT_MIPS_RLD_MAP, Out<ELFT>::MipsRldMap});
  }

  // +1 for DT_NULL
  Header.sh_size = (Entries.size() + 1) * Header.sh_entsize;
}

template <class ELFT> void DynamicSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

  for (const Entry &E : Entries) {
    P->d_tag = E.Tag;
    switch (E.Kind) {
    case Entry::SecAddr:
      P->d_un.d_ptr = E.OutSec->getVA();
      break;
    case Entry::SymAddr:
      P->d_un.d_ptr = getSymVA<ELFT>(*E.Sym);
      break;
    case Entry::PlainInt:
      P->d_un.d_val = E.Val;
      break;
    }
    ++P;
  }
}

template <class ELFT>
EhFrameHeader<ELFT>::EhFrameHeader()
    : OutputSectionBase<ELFT>(".eh_frame_hdr", llvm::ELF::SHT_PROGBITS,
                              SHF_ALLOC) {
  // It's a 4 bytes of header + pointer to the contents of the .eh_frame section
  // + the number of FDE pointers in the table.
  this->Header.sh_size = 12;
}

// We have to get PC values of FDEs. They depend on relocations
// which are target specific, so we run this code after performing
// all relocations. We read the values from ouput buffer according to the
// encoding given for FDEs. Return value is an offset to the initial PC value
// for the FDE.
template <class ELFT>
typename EhFrameHeader<ELFT>::uintX_t
EhFrameHeader<ELFT>::getFdePc(uintX_t EhVA, const FdeData &F) {
  const endianness E = ELFT::TargetEndianness;
  assert((F.Enc & 0xF0) != dwarf::DW_EH_PE_datarel);

  uintX_t FdeOff = EhVA + F.Off + 8;
  switch (F.Enc & 0xF) {
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return FdeOff + read16<E>(F.PCRel);
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return FdeOff + read32<E>(F.PCRel);
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return FdeOff + read64<E>(F.PCRel);
  case dwarf::DW_EH_PE_absptr:
    if (sizeof(uintX_t) == 8)
      return FdeOff + read64<E>(F.PCRel);
    return FdeOff + read32<E>(F.PCRel);
  }
  fatal("unknown FDE size encoding");
}

template <class ELFT> void EhFrameHeader<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;

  const uint8_t Header[] = {1, dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4,
                            dwarf::DW_EH_PE_udata4,
                            dwarf::DW_EH_PE_datarel | dwarf::DW_EH_PE_sdata4};
  memcpy(Buf, Header, sizeof(Header));

  uintX_t EhVA = Sec->getVA();
  uintX_t VA = this->getVA();
  uintX_t EhOff = EhVA - VA - 4;
  write32<E>(Buf + 4, EhOff);
  write32<E>(Buf + 8, this->FdeList.size());
  Buf += 12;

  // InitialPC -> Offset in .eh_frame, sorted by InitialPC.
  std::map<uintX_t, size_t> PcToOffset;
  for (const FdeData &F : FdeList)
    PcToOffset[getFdePc(EhVA, F)] = F.Off;

  for (auto &I : PcToOffset) {
    // The first four bytes are an offset to the initial PC value for the FDE.
    write32<E>(Buf, I.first - VA);
    // The last four bytes are an offset to the FDE data itself.
    write32<E>(Buf + 4, EhVA + I.second - VA);
    Buf += 8;
  }
}

template <class ELFT>
void EhFrameHeader<ELFT>::assignEhFrame(EHOutputSection<ELFT> *Sec) {
  assert((!this->Sec || this->Sec == Sec) &&
         "multiple .eh_frame sections not supported for .eh_frame_hdr");
  Live = Config->EhFrameHdr;
  this->Sec = Sec;
}

template <class ELFT>
void EhFrameHeader<ELFT>::addFde(uint8_t Enc, size_t Off, uint8_t *PCRel) {
  if (Live && (Enc & 0xF0) == dwarf::DW_EH_PE_datarel)
    fatal("DW_EH_PE_datarel encoding unsupported for FDEs by .eh_frame_hdr");
  FdeList.push_back(FdeData{Enc, Off, PCRel});
}

template <class ELFT> void EhFrameHeader<ELFT>::reserveFde() {
  // Each FDE entry is 8 bytes long:
  // The first four bytes are an offset to the initial PC value for the FDE. The
  // last four byte are an offset to the FDE data itself.
  this->Header.sh_size += 8;
}

template <class ELFT>
OutputSection<ELFT>::OutputSection(StringRef Name, uint32_t Type,
                                   uintX_t Flags)
    : OutputSectionBase<ELFT>(Name, Type, Flags) {}

template <class ELFT>
void OutputSection<ELFT>::addSection(InputSectionBase<ELFT> *C) {
  auto *S = cast<InputSection<ELFT>>(C);
  Sections.push_back(S);
  S->OutSec = this;
  uint32_t Align = S->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  uintX_t Off = this->Header.sh_size;
  Off = alignTo(Off, Align);
  S->OutSecOff = Off;
  Off += S->getSize();
  this->Header.sh_size = Off;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t elf2::getSymVA(const SymbolBody &S) {
  switch (S.kind()) {
  case SymbolBody::DefinedSyntheticKind: {
    auto &D = cast<DefinedSynthetic<ELFT>>(S);
    return D.Section.getVA() + D.Value;
  }
  case SymbolBody::DefinedRegularKind: {
    const auto &DR = cast<DefinedRegular<ELFT>>(S);
    InputSectionBase<ELFT> *SC = DR.Section;
    if (!SC)
      return DR.Sym.st_value;

    // Symbol offsets for AMDGPU need to be the offset in bytes of the symbol
    // from the beginning of the section.
    if (Config->EMachine == EM_AMDGPU)
      return SC->getOffset(DR.Sym);
    if (DR.Sym.getType() == STT_TLS)
      return SC->OutSec->getVA() + SC->getOffset(DR.Sym) -
             Out<ELFT>::TlsPhdr->p_vaddr;
    return SC->OutSec->getVA() + SC->getOffset(DR.Sym);
  }
  case SymbolBody::DefinedCommonKind:
    return Out<ELFT>::Bss->getVA() + cast<DefinedCommon>(S).OffsetInBss;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(S);
    if (SS.NeedsCopy)
      return Out<ELFT>::Bss->getVA() + SS.OffsetInBss;
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
elf2::getLocalRelTarget(const ObjectFile<ELFT> &File,
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
    fatal("Unsupported relocation without symbol");

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
  return VA + Section->getOffset(Offset) + Addend;
}

// Returns true if a symbol can be replaced at load-time by a symbol
// with the same name defined in other ELF executable or DSO.
bool elf2::canBePreempted(const SymbolBody *Body, bool NeedsGot) {
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
EHOutputSection<ELFT>::EHOutputSection(StringRef Name, uint32_t Type,
                                       uintX_t Flags)
    : OutputSectionBase<ELFT>(Name, Type, Flags) {
  Out<ELFT>::EhFrameHdr->assignEhFrame(this);
}

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

// Read a byte and advance D by one byte.
static uint8_t readByte(ArrayRef<uint8_t> &D) {
  if (D.empty())
    fatal("corrupted or unsupported CIE information");
  uint8_t B = D.front();
  D = D.slice(1);
  return B;
}

static void skipLeb128(ArrayRef<uint8_t> &D) {
  while (!D.empty()) {
    uint8_t Val = D.front();
    D = D.slice(1);
    if ((Val & 0x80) == 0)
      return;
  }
  fatal("corrupted or unsupported CIE information");
}

template <class ELFT> static unsigned getSizeForEncoding(unsigned Enc) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  switch (Enc & 0x0f) {
  default:
    fatal("unknown FDE encoding");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return sizeof(uintX_t);
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }
}

template <class ELFT>
uint8_t EHOutputSection<ELFT>::getFdeEncoding(ArrayRef<uint8_t> D) {
  auto Check = [](bool C) {
    if (!C)
      fatal("corrupted or unsupported CIE information");
  };

  Check(D.size() >= 8);
  D = D.slice(8);

  uint8_t Version = readByte(D);
  if (Version != 1 && Version != 3)
    fatal("FDE version 1 or 3 expected, but got " + Twine((unsigned)Version));

  auto AugEnd = std::find(D.begin() + 1, D.end(), '\0');
  Check(AugEnd != D.end());
  ArrayRef<uint8_t> AugString(D.begin(), AugEnd - D.begin());
  D = D.slice(AugString.size() + 1);

  // Code alignment factor should always be 1 for .eh_frame.
  if (readByte(D) != 1)
    fatal("CIE code alignment must be 1");
  // Skip data alignment factor
  skipLeb128(D);

  // Skip the return address register. In CIE version 1 this is a single
  // byte. In CIE version 3 this is an unsigned LEB128.
  if (Version == 1)
    readByte(D);
  else
    skipLeb128(D);

  while (!AugString.empty()) {
    switch (readByte(AugString)) {
    case 'z':
      skipLeb128(D);
      break;
    case 'R':
      return readByte(D);
    case 'P': {
      uint8_t Enc = readByte(D);
      if ((Enc & 0xf0) == dwarf::DW_EH_PE_aligned)
        fatal("DW_EH_PE_aligned encoding for address of a personality routine "
              "handler not supported");
      unsigned EncSize = getSizeForEncoding<ELFT>(Enc);
      Check(D.size() >= EncSize);
      D = D.slice(EncSize);
      break;
    }
    case 'S':
    case 'L':
      // L: Language Specific Data Area (LSDA) encoding
      // S: This CIE represents a stack frame for the invocation of a signal
      //    handler
      break;
    default:
      fatal("unknown .eh_frame augmentation string value");
    }
  }
  return dwarf::DW_EH_PE_absptr;
}

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
    // If CIE/FDE data length is zero then Length is 4, this
    // shall be considered a terminator and processing shall end.
    if (Length == 4)
      break;
    StringRef Entry((const char *)D.data(), Length);

    while (RelI != RelE && RelI->r_offset < Offset)
      ++RelI;
    uintX_t NextOffset = Offset + Length;
    bool HasReloc = RelI != RelE && RelI->r_offset < NextOffset;

    uint32_t ID = read32<E>(D.data() + 4);
    if (ID == 0) {
      // CIE
      Cie<ELFT> C(S, Index);
      if (Config->EhFrameHdr)
        C.FdeEncoding = getFdeEncoding(D);

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
        this->Header.sh_size += alignTo(Length, sizeof(uintX_t));
      }
      OffsetToIndex[Offset] = P.first->second;
    } else {
      if (!HasReloc)
        fatal("FDE doesn't reference another section");
      InputSectionBase<ELFT> *Target = S->getRelocTarget(*RelI);
      if (Target != &InputSection<ELFT>::Discarded && Target->isLive()) {
        uint32_t CieOffset = Offset + 4 - ID;
        auto I = OffsetToIndex.find(CieOffset);
        if (I == OffsetToIndex.end())
          fatal("Invalid CIE reference");
        Cies[I->second].Fdes.push_back(EHRegion<ELFT>(S, Index));
        Out<ELFT>::EhFrameHdr->reserveFde();
        this->Header.sh_size += alignTo(Length, sizeof(uintX_t));
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
    fatal("Truncated CIE/FDE length");
  uint64_t Len = read32<E>(D.data());
  if (Len < UINT32_MAX) {
    if (Len > (UINT32_MAX - 4))
      fatal("CIE/FIE size is too large");
    if (Len + 4 > D.size())
      fatal("CIE/FIE ends past the end of the section");
    return Len + 4;
  }

  if (D.size() < 12)
    fatal("Truncated CIE/FDE length");
  Len = read64<E>(D.data() + 4);
  if (Len > (UINT64_MAX - 12))
    fatal("CIE/FIE size is too large");
  if (Len + 12 > D.size())
    fatal("CIE/FIE ends past the end of the section");
  return Len + 12;
}

template <class ELFT>
void EHOutputSection<ELFT>::addSection(InputSectionBase<ELFT> *C) {
  auto *S = cast<EHInputSection<ELFT>>(C);
  const Elf_Shdr *RelSec = S->RelocSection;
  if (!RelSec) {
    addSectionAux(S, make_range<const Elf_Rela *>(nullptr, nullptr));
    return;
  }
  ELFFile<ELFT> &Obj = S->getFile()->getObj();
  if (RelSec->sh_type == SHT_RELA)
    addSectionAux(S, Obj.relas(RelSec));
  else
    addSectionAux(S, Obj.rels(RelSec));
}

template <class ELFT>
static typename ELFFile<ELFT>::uintX_t writeAlignedCieOrFde(StringRef Data,
                                                            uint8_t *Buf) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  const endianness E = ELFT::TargetEndianness;
  uint64_t Len = alignTo(Data.size(), sizeof(uintX_t));
  write32<E>(Buf, Len - 4);
  memcpy(Buf + 4, Data.data() + 4, Data.size() - 4);
  return Len;
}

template <class ELFT> void EHOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;
  size_t Offset = 0;
  for (const Cie<ELFT> &C : Cies) {
    size_t CieOffset = Offset;

    uintX_t CIELen = writeAlignedCieOrFde<ELFT>(C.data(), Buf + Offset);
    C.S->Offsets[C.Index].second = Offset;
    Offset += CIELen;

    for (const EHRegion<ELFT> &F : C.Fdes) {
      uintX_t Len = writeAlignedCieOrFde<ELFT>(F.data(), Buf + Offset);
      write32<E>(Buf + Offset + 4, Offset + 4 - CieOffset); // Pointer
      F.S->Offsets[F.Index].second = Offset;
      Out<ELFT>::EhFrameHdr->addFde(C.FdeEncoding, Offset, Buf + Offset + 8);
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
MergeOutputSection<ELFT>::MergeOutputSection(StringRef Name, uint32_t Type,
                                             uintX_t Flags)
    : OutputSectionBase<ELFT>(Name, Type, Flags) {}

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
void MergeOutputSection<ELFT>::addSection(InputSectionBase<ELFT> *C) {
  auto *S = cast<MergeInputSection<ELFT>>(C);
  S->OutSec = this;
  uint32_t Align = S->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  ArrayRef<uint8_t> D = S->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  uintX_t EntSize = S->getSectionHdr()->sh_entsize;

  if (this->Header.sh_flags & SHF_STRINGS) {
    uintX_t Offset = 0;
    while (!Data.empty()) {
      size_t End = findNull(Data, EntSize);
      if (End == StringRef::npos)
        fatal("String is not null terminated");
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
      S->Offsets.push_back(std::make_pair(I, OutputOffset));
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
    : OutputSectionBase<ELFT>(Name, SHT_STRTAB,
                              Dynamic ? (uintX_t)SHF_ALLOC : 0),
      Dynamic(Dynamic) {
  this->Header.sh_addralign = 1;
}

// String tables are created in two phases. First you call reserve()
// to reserve room in the string table, and then call addString() to actually
// add that string.
//
// Why two phases? We want to know the size of the string table as early as
// possible to fix file layout. So we have separated finalize(), which
// determines the size of the section, from writeTo(), which writes the section
// contents to the output buffer. If we merge reserve() with addString(),
// we need a plumbing work for finalize() and writeTo() so that offsets
// we obtained in the former function can be written in the latter.
// This design eliminated that need.
template <class ELFT> void StringTableSection<ELFT>::reserve(StringRef S) {
  Reserved += S.size() + 1; // +1 for NUL
}

// Adds a string to the string table. You must call reserve() with the
// same string before calling addString().
template <class ELFT> size_t StringTableSection<ELFT>::addString(StringRef S) {
  size_t Pos = Used;
  Strings.push_back(S);
  Used += S.size() + 1;
  Reserved -= S.size() + 1;
  assert((int64_t)Reserved >= 0);
  return Pos;
}

template <class ELFT> void StringTableSection<ELFT>::writeTo(uint8_t *Buf) {
  // ELF string tables start with NUL byte, so advance the pointer by one.
  ++Buf;
  for (StringRef S : Strings) {
    memcpy(Buf, S.data(), S.size());
    Buf += S.size() + 1;
  }
}

template <class ELFT>
SymbolTableSection<ELFT>::SymbolTableSection(
    SymbolTable<ELFT> &Table, StringTableSection<ELFT> &StrTabSec)
    : OutputSectionBase<ELFT>(StrTabSec.isDynamic() ? ".dynsym" : ".symtab",
                              StrTabSec.isDynamic() ? SHT_DYNSYM : SHT_SYMTAB,
                              StrTabSec.isDynamic() ? (uintX_t)SHF_ALLOC : 0),
      Table(Table), StrTabSec(StrTabSec) {
  this->Header.sh_entsize = sizeof(Elf_Sym);
  this->Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
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
  StrTabSec.reserve(Name);
  ++NumLocals;
}

template <class ELFT>
void SymbolTableSection<ELFT>::addSymbol(SymbolBody *Body) {
  StrTabSec.reserve(Body->getName());
  Symbols.push_back(Body);
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
    for (const Elf_Sym *Sym : File->KeptLocalSyms) {
      ErrorOr<StringRef> SymNameOrErr = Sym->getName(File->getStringTable());
      fatal(SymNameOrErr);
      StringRef SymName = *SymNameOrErr;

      auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
      uintX_t VA = 0;
      if (Sym->st_shndx == SHN_ABS) {
        ESym->st_shndx = SHN_ABS;
        VA = Sym->st_value;
      } else {
        InputSectionBase<ELFT> *Section = File->getSection(*Sym);
        const OutputSectionBase<ELFT> *OutSec = Section->OutSec;
        ESym->st_shndx = OutSec->SectionIndex;
        VA = Section->getOffset(*Sym);
        // Symbol offsets for AMDGPU need to be the offset in bytes of the
        // symbol from the beginning of the section.
        if (Config->EMachine != EM_AMDGPU)
          VA += OutSec->getVA();
      }
      ESym->st_name = StrTabSec.addString(SymName);
      ESym->st_size = Sym->st_size;
      ESym->setBindingAndType(Sym->getBinding(), Sym->getType());
      ESym->st_value = VA;
      Buf += sizeof(*ESym);
    }
  }
}

template <class ELFT>
static const typename llvm::object::ELFFile<ELFT>::Elf_Sym *
getElfSym(SymbolBody &Body) {
  if (auto *EBody = dyn_cast<DefinedElf<ELFT>>(&Body))
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
      if (InputSectionBase<ELFT> *Sec = Sym->Section) {
        if (!Sec->isLive())
          continue;
        OutSec = Sec->OutSec;
      }
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
    case SymbolBody::LazyKind:
      break;
    }

    StringRef Name = Body->getName();
    ESym->st_name = StrTabSec.addString(Name);

    unsigned char Type = STT_NOTYPE;
    uintX_t Size = 0;
    if (const Elf_Sym *InputSym = getElfSym<ELFT>(*Body)) {
      Type = InputSym->getType();
      Size = InputSym->st_size;
    } else if (auto *C = dyn_cast<DefinedCommon>(Body)) {
      Type = STT_OBJECT;
      Size = C->Size;
    }

    ESym->setBindingAndType(getSymbolBinding(Body), Type);
    ESym->st_size = Size;
    ESym->setVisibility(Body->getVisibility());
    ESym->st_value = getSymVA<ELFT>(*Body);

    if (OutSec)
      ESym->st_shndx = OutSec->SectionIndex;
    else if (isa<DefinedRegular<ELFT>>(Body))
      ESym->st_shndx = SHN_ABS;

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
  if (isa<DefinedSynthetic<ELFT>>(Body))
    return STB_LOCAL;
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
  R->ri_gprmask = GprMask;
}

template <class ELFT>
void MipsReginfoOutputSection<ELFT>::addSection(InputSectionBase<ELFT> *C) {
  // Copy input object file's .reginfo gprmask to output.
  auto *S = cast<MipsReginfoInputSection<ELFT>>(C);
  GprMask |= S->Reginfo->ri_gprmask;
}

namespace lld {
namespace elf2 {
template class OutputSectionBase<ELF32LE>;
template class OutputSectionBase<ELF32BE>;
template class OutputSectionBase<ELF64LE>;
template class OutputSectionBase<ELF64BE>;

template class EhFrameHeader<ELF32LE>;
template class EhFrameHeader<ELF32BE>;
template class EhFrameHeader<ELF64LE>;
template class EhFrameHeader<ELF64BE>;

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

template uint32_t getLocalRelTarget(const ObjectFile<ELF32LE> &,
                                    const ELFFile<ELF32LE>::Elf_Rel &,
                                    uint32_t);
template uint32_t getLocalRelTarget(const ObjectFile<ELF32BE> &,
                                    const ELFFile<ELF32BE>::Elf_Rel &,
                                    uint32_t);
template uint64_t getLocalRelTarget(const ObjectFile<ELF64LE> &,
                                    const ELFFile<ELF64LE>::Elf_Rel &,
                                    uint64_t);
template uint64_t getLocalRelTarget(const ObjectFile<ELF64BE> &,
                                    const ELFFile<ELF64BE>::Elf_Rel &,
                                    uint64_t);
template uint32_t getLocalRelTarget(const ObjectFile<ELF32LE> &,
                                    const ELFFile<ELF32LE>::Elf_Rela &,
                                    uint32_t);
template uint32_t getLocalRelTarget(const ObjectFile<ELF32BE> &,
                                    const ELFFile<ELF32BE>::Elf_Rela &,
                                    uint32_t);
template uint64_t getLocalRelTarget(const ObjectFile<ELF64LE> &,
                                    const ELFFile<ELF64LE>::Elf_Rela &,
                                    uint64_t);
template uint64_t getLocalRelTarget(const ObjectFile<ELF64BE> &,
                                    const ELFFile<ELF64BE>::Elf_Rela &,
                                    uint64_t);
}
}
