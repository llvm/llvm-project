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
#include "EhFrame.h"
#include "GdbIndex.h"
#include "LinkerScript.h"
#include "Memory.h"
#include "Strings.h"
#include "SymbolListFile.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Core/Parallel.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA1.h"

using namespace llvm;
using namespace llvm::dwarf;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

OutputSectionBase::OutputSectionBase(StringRef Name, uint32_t Type,
                                     uint64_t Flags)
    : Name(Name) {
  this->Type = Type;
  this->Flags = Flags;
  this->Addralign = 1;
}

uint32_t OutputSectionBase::getPhdrFlags() const {
  uint32_t Ret = PF_R;
  if (Flags & SHF_WRITE)
    Ret |= PF_W;
  if (Flags & SHF_EXECINSTR)
    Ret |= PF_X;
  return Ret;
}

template <class ELFT>
void OutputSectionBase::writeHeaderTo(typename ELFT::Shdr *Shdr) {
  Shdr->sh_entsize = Entsize;
  Shdr->sh_addralign = Addralign;
  Shdr->sh_type = Type;
  Shdr->sh_offset = Offset;
  Shdr->sh_flags = Flags;
  Shdr->sh_info = Info;
  Shdr->sh_link = Link;
  Shdr->sh_addr = Addr;
  Shdr->sh_size = Size;
  Shdr->sh_name = ShName;
}

template <class ELFT>
GdbIndexSection<ELFT>::GdbIndexSection()
    : OutputSectionBase(".gdb_index", SHT_PROGBITS, 0) {}

template <class ELFT> void GdbIndexSection<ELFT>::parseDebugSections() {
  std::vector<InputSection<ELFT> *> &IS =
      static_cast<OutputSection<ELFT> *>(Out<ELFT>::DebugInfo)->Sections;

  for (InputSection<ELFT> *I : IS)
    readDwarf(I);
}

template <class ELFT>
void GdbIndexSection<ELFT>::readDwarf(InputSection<ELFT> *I) {
  std::vector<std::pair<uintX_t, uintX_t>> CuList = readCuList(I);
  CompilationUnits.insert(CompilationUnits.end(), CuList.begin(), CuList.end());
}

template <class ELFT> void GdbIndexSection<ELFT>::finalize() {
  parseDebugSections();

  // GdbIndex header consist from version fields
  // and 5 more fields with different kinds of offsets.
  CuTypesOffset = CuListOffset + CompilationUnits.size() * CompilationUnitSize;
  this->Size = CuTypesOffset;
}

template <class ELFT> void GdbIndexSection<ELFT>::writeTo(uint8_t *Buf) {
  write32le(Buf, 7);                  // Write Version
  write32le(Buf + 4, CuListOffset);   // CU list offset
  write32le(Buf + 8, CuTypesOffset);  // Types CU list offset
  write32le(Buf + 12, CuTypesOffset); // Address area offset
  write32le(Buf + 16, CuTypesOffset); // Symbol table offset
  write32le(Buf + 20, CuTypesOffset); // Constant pool offset
  Buf += 24;

  // Write the CU list.
  for (std::pair<uintX_t, uintX_t> CU : CompilationUnits) {
    write64le(Buf, CU.first);
    write64le(Buf + 8, CU.second);
    Buf += 16;
  }
}

template <class ELFT>
PltSection<ELFT>::PltSection()
    : OutputSectionBase(".plt", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR) {
  this->Addralign = 16;
}

template <class ELFT> void PltSection<ELFT>::writeTo(uint8_t *Buf) {
  // At beginning of PLT, we have code to call the dynamic linker
  // to resolve dynsyms at runtime. Write such code.
  Target->writePltHeader(Buf);
  size_t Off = Target->PltHeaderSize;

  for (auto &I : Entries) {
    const SymbolBody *B = I.first;
    unsigned RelOff = I.second;
    uint64_t Got = B->getGotPltVA<ELFT>();
    uint64_t Plt = this->Addr + Off;
    Target->writePlt(Buf + Off, Got, Plt, B->PltIndex, RelOff);
    Off += Target->PltEntrySize;
  }
}

template <class ELFT> void PltSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.PltIndex = Entries.size();
  unsigned RelOff = In<ELFT>::RelaPlt->getRelocOffset();
  Entries.push_back(std::make_pair(&Sym, RelOff));
}

template <class ELFT> void PltSection<ELFT>::finalize() {
  this->Size = Target->PltHeaderSize + Entries.size() * Target->PltEntrySize;
}

template <class ELFT>
HashTableSection<ELFT>::HashTableSection()
    : OutputSectionBase(".hash", SHT_HASH, SHF_ALLOC) {
  this->Entsize = sizeof(Elf_Word);
  this->Addralign = sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::finalize() {
  this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;

  unsigned NumEntries = 2;                             // nbucket and nchain.
  NumEntries += In<ELFT>::DynSymTab->getNumSymbols();  // The chain entries.

  // Create as many buckets as there are symbols.
  // FIXME: This is simplistic. We can try to optimize it, but implementing
  // support for SHT_GNU_HASH is probably even more profitable.
  NumEntries += In<ELFT>::DynSymTab->getNumSymbols();
  this->Size = NumEntries * sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  unsigned NumSymbols = In<ELFT>::DynSymTab->getNumSymbols();
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NumSymbols; // nbucket
  *P++ = NumSymbols; // nchain

  Elf_Word *Buckets = P;
  Elf_Word *Chains = P + NumSymbols;

  for (const SymbolTableEntry &S : In<ELFT>::DynSymTab->getSymbols()) {
    SymbolBody *Body = S.Symbol;
    StringRef Name = Body->getName();
    unsigned I = Body->DynsymIndex;
    uint32_t Hash = hashSysV(Name) % NumSymbols;
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
    : OutputSectionBase(".gnu.hash", SHT_GNU_HASH, SHF_ALLOC) {
  this->Entsize = ELFT::Is64Bits ? 0 : 4;
  this->Addralign = sizeof(uintX_t);
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
  unsigned NumHashed = Symbols.size();
  NBuckets = calcNBuckets(NumHashed);
  MaskWords = calcMaskWords(NumHashed);
  // Second hash shift estimation: just predefined values.
  Shift2 = ELFT::Is64Bits ? 6 : 5;

  this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;
  this->Size = sizeof(Elf_Word) * 4            // Header
               + sizeof(Elf_Off) * MaskWords   // Bloom Filter
               + sizeof(Elf_Word) * NBuckets   // Hash Buckets
               + sizeof(Elf_Word) * NumHashed; // Hash Values
}

template <class ELFT> void GnuHashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  writeHeader(Buf);
  if (Symbols.empty())
    return;
  writeBloomFilter(Buf);
  writeHashTable(Buf);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeHeader(uint8_t *&Buf) {
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NBuckets;
  *P++ = In<ELFT>::DynSymTab->getNumSymbols() - Symbols.size();
  *P++ = MaskWords;
  *P++ = Shift2;
  Buf = reinterpret_cast<uint8_t *>(P);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeBloomFilter(uint8_t *&Buf) {
  unsigned C = sizeof(Elf_Off) * 8;

  auto *Masks = reinterpret_cast<Elf_Off *>(Buf);
  for (const SymbolData &Sym : Symbols) {
    size_t Pos = (Sym.Hash / C) & (MaskWords - 1);
    uintX_t V = (uintX_t(1) << (Sym.Hash % C)) |
                (uintX_t(1) << ((Sym.Hash >> Shift2) % C));
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
  for (const SymbolData &Sym : Symbols) {
    int Bucket = Sym.Hash % NBuckets;
    assert(PrevBucket <= Bucket);
    if (Bucket != PrevBucket) {
      Buckets[Bucket] = Sym.Body->DynsymIndex;
      PrevBucket = Bucket;
      if (I > 0)
        Values[I - 1] |= 1;
    }
    Values[I] = Sym.Hash & ~1;
    ++I;
  }
  if (I > 0)
    Values[I - 1] |= 1;
}

// Add symbols to this symbol hash table. Note that this function
// destructively sort a given vector -- which is needed because
// GNU-style hash table places some sorting requirements.
template <class ELFT>
void GnuHashTableSection<ELFT>::addSymbols(std::vector<SymbolTableEntry> &V) {
  // Ideally this will just be 'auto' but GCC 6.1 is not able
  // to deduce it correctly.
  std::vector<SymbolTableEntry>::iterator Mid =
      std::stable_partition(V.begin(), V.end(), [](const SymbolTableEntry &S) {
        return S.Symbol->isUndefined();
      });
  if (Mid == V.end())
    return;
  for (auto I = Mid, E = V.end(); I != E; ++I) {
    SymbolBody *B = I->Symbol;
    size_t StrOff = I->StrTabOffset;
    Symbols.push_back({B, StrOff, hashGnu(B->getName())});
  }

  unsigned NBuckets = calcNBuckets(Symbols.size());
  std::stable_sort(Symbols.begin(), Symbols.end(),
                   [&](const SymbolData &L, const SymbolData &R) {
                     return L.Hash % NBuckets < R.Hash % NBuckets;
                   });

  V.erase(Mid, V.end());
  for (const SymbolData &Sym : Symbols)
    V.push_back({Sym.Body, Sym.STName});
}

// Returns the number of version definition entries. Because the first entry
// is for the version definition itself, it is the number of versioned symbols
// plus one. Note that we don't support multiple versions yet.
static unsigned getVerDefNum() { return Config->VersionDefinitions.size() + 1; }

template <class ELFT>
EhFrameHeader<ELFT>::EhFrameHeader()
    : OutputSectionBase(".eh_frame_hdr", SHT_PROGBITS, SHF_ALLOC) {}

// .eh_frame_hdr contains a binary search table of pointers to FDEs.
// Each entry of the search table consists of two values,
// the starting PC from where FDEs covers, and the FDE's address.
// It is sorted by PC.
template <class ELFT> void EhFrameHeader<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;

  // Sort the FDE list by their PC and uniqueify. Usually there is only
  // one FDE for a PC (i.e. function), but if ICF merges two functions
  // into one, there can be more than one FDEs pointing to the address.
  auto Less = [](const FdeData &A, const FdeData &B) { return A.Pc < B.Pc; };
  std::stable_sort(Fdes.begin(), Fdes.end(), Less);
  auto Eq = [](const FdeData &A, const FdeData &B) { return A.Pc == B.Pc; };
  Fdes.erase(std::unique(Fdes.begin(), Fdes.end(), Eq), Fdes.end());

  Buf[0] = 1;
  Buf[1] = DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  Buf[2] = DW_EH_PE_udata4;
  Buf[3] = DW_EH_PE_datarel | DW_EH_PE_sdata4;
  write32<E>(Buf + 4, Out<ELFT>::EhFrame->Addr - this->Addr - 4);
  write32<E>(Buf + 8, Fdes.size());
  Buf += 12;

  uintX_t VA = this->Addr;
  for (FdeData &Fde : Fdes) {
    write32<E>(Buf, Fde.Pc - VA);
    write32<E>(Buf + 4, Fde.FdeVA - VA);
    Buf += 8;
  }
}

template <class ELFT> void EhFrameHeader<ELFT>::finalize() {
  // .eh_frame_hdr has a 12 bytes header followed by an array of FDEs.
  this->Size = 12 + Out<ELFT>::EhFrame->NumFdes * 8;
}

template <class ELFT>
void EhFrameHeader<ELFT>::addFde(uint32_t Pc, uint32_t FdeVA) {
  Fdes.push_back({Pc, FdeVA});
}

template <class ELFT> static uint64_t getEntsize(uint32_t Type) {
  switch (Type) {
  case SHT_RELA:
    return sizeof(typename ELFT::Rela);
  case SHT_REL:
    return sizeof(typename ELFT::Rel);
  case SHT_MIPS_REGINFO:
    return sizeof(Elf_Mips_RegInfo<ELFT>);
  case SHT_MIPS_OPTIONS:
    return sizeof(Elf_Mips_Options<ELFT>) + sizeof(Elf_Mips_RegInfo<ELFT>);
  case SHT_MIPS_ABIFLAGS:
    return sizeof(Elf_Mips_ABIFlags<ELFT>);
  default:
    return 0;
  }
}

template <class ELFT>
OutputSection<ELFT>::OutputSection(StringRef Name, uint32_t Type, uintX_t Flags)
    : OutputSectionBase(Name, Type, Flags) {
  this->Entsize = getEntsize<ELFT>(Type);
}

template <class ELFT> void OutputSection<ELFT>::finalize() {
  if ((this->Flags & SHF_LINK_ORDER) && !this->Sections.empty()) {
    // We must preserve the link order dependency of sections with the
    // SHF_LINK_ORDER flag. The dependency is indicated by the sh_link field. We
    // need to translate the InputSection sh_link to the OutputSection sh_link,
    // all InputSections in the OutputSection have the same dependency.
    if (auto *D = this->Sections.front()->getLinkOrderDep())
      this->Link = D->OutSec->SectionIndex;
  }

  uint32_t Type = this->Type;
  if (!Config->Relocatable || (Type != SHT_RELA && Type != SHT_REL))
    return;

  this->Link = In<ELFT>::SymTab->OutSec->SectionIndex;
  // sh_info for SHT_REL[A] sections should contain the section header index of
  // the section to which the relocation applies.
  InputSectionBase<ELFT> *S = Sections[0]->getRelocatedSection();
  this->Info = S->OutSec->SectionIndex;
}

template <class ELFT>
void OutputSection<ELFT>::addSection(InputSectionData *C) {
  assert(C->Live);
  auto *S = cast<InputSection<ELFT>>(C);
  Sections.push_back(S);
  S->OutSec = this;
  this->updateAlignment(S->Alignment);
  // Keep sh_entsize value of the input section to be able to perform merging
  // later during a final linking using the generated relocatable object.
  if (Config->Relocatable && (S->Flags & SHF_MERGE))
    this->Entsize = S->Entsize;
}

// This function is called after we sort input sections
// and scan relocations to setup sections' offsets.
template <class ELFT> void OutputSection<ELFT>::assignOffsets() {
  uintX_t Off = this->Size;
  for (InputSection<ELFT> *S : Sections) {
    Off = alignTo(Off, S->Alignment);
    S->OutSecOff = Off;
    Off += S->getSize();
  }
  this->Size = Off;
}

template <class ELFT>
void OutputSection<ELFT>::sort(
    std::function<unsigned(InputSection<ELFT> *S)> Order) {
  typedef std::pair<unsigned, InputSection<ELFT> *> Pair;
  auto Comp = [](const Pair &A, const Pair &B) { return A.first < B.first; };

  std::vector<Pair> V;
  for (InputSection<ELFT> *S : Sections)
    V.push_back({Order(S), S});
  std::stable_sort(V.begin(), V.end(), Comp);
  Sections.clear();
  for (Pair &P : V)
    Sections.push_back(P.second);
}

// Sorts input sections by section name suffixes, so that .foo.N comes
// before .foo.M if N < M. Used to sort .{init,fini}_array.N sections.
// We want to keep the original order if the priorities are the same
// because the compiler keeps the original initialization order in a
// translation unit and we need to respect that.
// For more detail, read the section of the GCC's manual about init_priority.
template <class ELFT> void OutputSection<ELFT>::sortInitFini() {
  // Sort sections by priority.
  sort([](InputSection<ELFT> *S) { return getPriority(S->Name); });
}

// Returns true if S matches /Filename.?\.o$/.
static bool isCrtBeginEnd(StringRef S, StringRef Filename) {
  if (!S.endswith(".o"))
    return false;
  S = S.drop_back(2);
  if (S.endswith(Filename))
    return true;
  return !S.empty() && S.drop_back().endswith(Filename);
}

static bool isCrtbegin(StringRef S) { return isCrtBeginEnd(S, "crtbegin"); }
static bool isCrtend(StringRef S) { return isCrtBeginEnd(S, "crtend"); }

// .ctors and .dtors are sorted by this priority from highest to lowest.
//
//  1. The section was contained in crtbegin (crtbegin contains
//     some sentinel value in its .ctors and .dtors so that the runtime
//     can find the beginning of the sections.)
//
//  2. The section has an optional priority value in the form of ".ctors.N"
//     or ".dtors.N" where N is a number. Unlike .{init,fini}_array,
//     they are compared as string rather than number.
//
//  3. The section is just ".ctors" or ".dtors".
//
//  4. The section was contained in crtend, which contains an end marker.
//
// In an ideal world, we don't need this function because .init_array and
// .ctors are duplicate features (and .init_array is newer.) However, there
// are too many real-world use cases of .ctors, so we had no choice to
// support that with this rather ad-hoc semantics.
template <class ELFT>
static bool compCtors(const InputSection<ELFT> *A,
                      const InputSection<ELFT> *B) {
  bool BeginA = isCrtbegin(A->getFile()->getName());
  bool BeginB = isCrtbegin(B->getFile()->getName());
  if (BeginA != BeginB)
    return BeginA;
  bool EndA = isCrtend(A->getFile()->getName());
  bool EndB = isCrtend(B->getFile()->getName());
  if (EndA != EndB)
    return EndB;
  StringRef X = A->Name;
  StringRef Y = B->Name;
  assert(X.startswith(".ctors") || X.startswith(".dtors"));
  assert(Y.startswith(".ctors") || Y.startswith(".dtors"));
  X = X.substr(6);
  Y = Y.substr(6);
  if (X.empty() && Y.empty())
    return false;
  return X < Y;
}

// Sorts input sections by the special rules for .ctors and .dtors.
// Unfortunately, the rules are different from the one for .{init,fini}_array.
// Read the comment above.
template <class ELFT> void OutputSection<ELFT>::sortCtorsDtors() {
  std::stable_sort(Sections.begin(), Sections.end(), compCtors<ELFT>);
}

static void fill(uint8_t *Buf, size_t Size, ArrayRef<uint8_t> A) {
  size_t I = 0;
  for (; I + A.size() < Size; I += A.size())
    memcpy(Buf + I, A.data(), A.size());
  memcpy(Buf + I, A.data(), Size - I);
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  ArrayRef<uint8_t> Filler = Script<ELFT>::X->getFiller(this->Name);
  if (!Filler.empty())
    fill(Buf, this->Size, Filler);
  auto Fn = [=](InputSection<ELFT> *IS) { IS->writeTo(Buf); };
  if (Config->Threads)
    parallel_for_each(Sections.begin(), Sections.end(), Fn);
  else
    std::for_each(Sections.begin(), Sections.end(), Fn);
  // Linker scripts may have BYTE()-family commands with which you
  // can write arbitrary bytes to the output. Process them if any.
  Script<ELFT>::X->writeDataBytes(this->Name, Buf);
}

template <class ELFT>
EhOutputSection<ELFT>::EhOutputSection()
    : OutputSectionBase(".eh_frame", SHT_PROGBITS, SHF_ALLOC) {}

// Search for an existing CIE record or create a new one.
// CIE records from input object files are uniquified by their contents
// and where their relocations point to.
template <class ELFT>
template <class RelTy>
CieRecord *EhOutputSection<ELFT>::addCie(EhSectionPiece &Piece,
                                         EhInputSection<ELFT> *Sec,
                                         ArrayRef<RelTy> Rels) {
  const endianness E = ELFT::TargetEndianness;
  if (read32<E>(Piece.data().data() + 4) != 0)
    fatal("CIE expected at beginning of .eh_frame: " + Sec->Name);

  SymbolBody *Personality = nullptr;
  unsigned FirstRelI = Piece.FirstRelocation;
  if (FirstRelI != (unsigned)-1)
    Personality = &Sec->getFile()->getRelocTargetSym(Rels[FirstRelI]);

  // Search for an existing CIE by CIE contents/relocation target pair.
  CieRecord *Cie = &CieMap[{Piece.data(), Personality}];

  // If not found, create a new one.
  if (Cie->Piece == nullptr) {
    Cie->Piece = &Piece;
    Cies.push_back(Cie);
  }
  return Cie;
}

// There is one FDE per function. Returns true if a given FDE
// points to a live function.
template <class ELFT>
template <class RelTy>
bool EhOutputSection<ELFT>::isFdeLive(EhSectionPiece &Piece,
                                      EhInputSection<ELFT> *Sec,
                                      ArrayRef<RelTy> Rels) {
  unsigned FirstRelI = Piece.FirstRelocation;
  if (FirstRelI == (unsigned)-1)
    fatal("FDE doesn't reference another section");
  const RelTy &Rel = Rels[FirstRelI];
  SymbolBody &B = Sec->getFile()->getRelocTargetSym(Rel);
  auto *D = dyn_cast<DefinedRegular<ELFT>>(&B);
  if (!D || !D->Section)
    return false;
  InputSectionBase<ELFT> *Target = D->Section->Repl;
  return Target && Target->Live;
}

// .eh_frame is a sequence of CIE or FDE records. In general, there
// is one CIE record per input object file which is followed by
// a list of FDEs. This function searches an existing CIE or create a new
// one and associates FDEs to the CIE.
template <class ELFT>
template <class RelTy>
void EhOutputSection<ELFT>::addSectionAux(EhInputSection<ELFT> *Sec,
                                          ArrayRef<RelTy> Rels) {
  const endianness E = ELFT::TargetEndianness;

  DenseMap<size_t, CieRecord *> OffsetToCie;
  for (EhSectionPiece &Piece : Sec->Pieces) {
    // The empty record is the end marker.
    if (Piece.size() == 4)
      return;

    size_t Offset = Piece.InputOff;
    uint32_t ID = read32<E>(Piece.data().data() + 4);
    if (ID == 0) {
      OffsetToCie[Offset] = addCie(Piece, Sec, Rels);
      continue;
    }

    uint32_t CieOffset = Offset + 4 - ID;
    CieRecord *Cie = OffsetToCie[CieOffset];
    if (!Cie)
      fatal("invalid CIE reference");

    if (!isFdeLive(Piece, Sec, Rels))
      continue;
    Cie->FdePieces.push_back(&Piece);
    NumFdes++;
  }
}

template <class ELFT>
void EhOutputSection<ELFT>::addSection(InputSectionData *C) {
  auto *Sec = cast<EhInputSection<ELFT>>(C);
  Sec->OutSec = this;
  this->updateAlignment(Sec->Alignment);
  Sections.push_back(Sec);

  // .eh_frame is a sequence of CIE or FDE records. This function
  // splits it into pieces so that we can call
  // SplitInputSection::getSectionPiece on the section.
  Sec->split();
  if (Sec->Pieces.empty())
    return;

  if (Sec->NumRelocations) {
    if (Sec->AreRelocsRela)
      addSectionAux(Sec, Sec->relas());
    else
      addSectionAux(Sec, Sec->rels());
    return;
  }
  addSectionAux(Sec, makeArrayRef<Elf_Rela>(nullptr, nullptr));
}

template <class ELFT>
static void writeCieFde(uint8_t *Buf, ArrayRef<uint8_t> D) {
  memcpy(Buf, D.data(), D.size());

  // Fix the size field. -4 since size does not include the size field itself.
  const endianness E = ELFT::TargetEndianness;
  write32<E>(Buf, alignTo(D.size(), sizeof(typename ELFT::uint)) - 4);
}

template <class ELFT> void EhOutputSection<ELFT>::finalize() {
  if (this->Size)
    return; // Already finalized.

  size_t Off = 0;
  for (CieRecord *Cie : Cies) {
    Cie->Piece->OutputOff = Off;
    Off += alignTo(Cie->Piece->size(), sizeof(uintX_t));

    for (EhSectionPiece *Fde : Cie->FdePieces) {
      Fde->OutputOff = Off;
      Off += alignTo(Fde->size(), sizeof(uintX_t));
    }
  }
  this->Size = Off;
}

template <class ELFT> static uint64_t readFdeAddr(uint8_t *Buf, int Size) {
  const endianness E = ELFT::TargetEndianness;
  switch (Size) {
  case DW_EH_PE_udata2:
    return read16<E>(Buf);
  case DW_EH_PE_udata4:
    return read32<E>(Buf);
  case DW_EH_PE_udata8:
    return read64<E>(Buf);
  case DW_EH_PE_absptr:
    if (ELFT::Is64Bits)
      return read64<E>(Buf);
    return read32<E>(Buf);
  }
  fatal("unknown FDE size encoding");
}

// Returns the VA to which a given FDE (on a mmap'ed buffer) is applied to.
// We need it to create .eh_frame_hdr section.
template <class ELFT>
typename ELFT::uint EhOutputSection<ELFT>::getFdePc(uint8_t *Buf, size_t FdeOff,
                                                    uint8_t Enc) {
  // The starting address to which this FDE applies is
  // stored at FDE + 8 byte.
  size_t Off = FdeOff + 8;
  uint64_t Addr = readFdeAddr<ELFT>(Buf + Off, Enc & 0x7);
  if ((Enc & 0x70) == DW_EH_PE_absptr)
    return Addr;
  if ((Enc & 0x70) == DW_EH_PE_pcrel)
    return Addr + this->Addr + Off;
  fatal("unknown FDE size relative encoding");
}

template <class ELFT> void EhOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;
  for (CieRecord *Cie : Cies) {
    size_t CieOffset = Cie->Piece->OutputOff;
    writeCieFde<ELFT>(Buf + CieOffset, Cie->Piece->data());

    for (EhSectionPiece *Fde : Cie->FdePieces) {
      size_t Off = Fde->OutputOff;
      writeCieFde<ELFT>(Buf + Off, Fde->data());

      // FDE's second word should have the offset to an associated CIE.
      // Write it.
      write32<E>(Buf + Off + 4, Off + 4 - CieOffset);
    }
  }

  for (EhInputSection<ELFT> *S : Sections)
    S->relocate(Buf, nullptr);

  // Construct .eh_frame_hdr. .eh_frame_hdr is a binary search table
  // to get a FDE from an address to which FDE is applied. So here
  // we obtain two addresses and pass them to EhFrameHdr object.
  if (Out<ELFT>::EhFrameHdr) {
    for (CieRecord *Cie : Cies) {
      uint8_t Enc = getFdeEncoding<ELFT>(Cie->Piece->data());
      for (SectionPiece *Fde : Cie->FdePieces) {
        uintX_t Pc = getFdePc(Buf, Fde->OutputOff, Enc);
        uintX_t FdeVA = this->Addr + Fde->OutputOff;
        Out<ELFT>::EhFrameHdr->addFde(Pc, FdeVA);
      }
    }
  }
}

template <class ELFT>
MergeOutputSection<ELFT>::MergeOutputSection(StringRef Name, uint32_t Type,
                                             uintX_t Flags, uintX_t Alignment)
    : OutputSectionBase(Name, Type, Flags),
      Builder(StringTableBuilder::RAW, Alignment) {}

template <class ELFT> void MergeOutputSection<ELFT>::writeTo(uint8_t *Buf) {
  Builder.write(Buf);
}

template <class ELFT>
void MergeOutputSection<ELFT>::addSection(InputSectionData *C) {
  auto *Sec = cast<MergeInputSection<ELFT>>(C);
  Sec->OutSec = this;
  this->updateAlignment(Sec->Alignment);
  this->Entsize = Sec->Entsize;
  Sections.push_back(Sec);
}

template <class ELFT> bool MergeOutputSection<ELFT>::shouldTailMerge() const {
  return (this->Flags & SHF_STRINGS) && Config->Optimize >= 2;
}

template <class ELFT> void MergeOutputSection<ELFT>::finalize() {
  // Add all string pieces to the string table builder to create section
  // contents. If we are not tail-optimizing, offsets of strings are fixed
  // when they are added to the builder (string table builder contains a
  // hash table from strings to offsets), so we record them if available.
  for (MergeInputSection<ELFT> *Sec : Sections) {
    for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I) {
      if (!Sec->Pieces[I].Live)
        continue;
      uint32_t OutputOffset = Builder.add(Sec->getData(I));

      // Save the offset in the generated string table.
      if (!shouldTailMerge())
        Sec->Pieces[I].OutputOff = OutputOffset;
    }
  }

  // Fix the string table content. After this, the contents
  // will never change.
  if (shouldTailMerge())
    Builder.finalize();
  else
    Builder.finalizeInOrder();
  this->Size = Builder.getSize();

  // finalize() fixed tail-optimized strings, so we can now get
  // offsets of strings. Get an offset for each string and save it
  // to a corresponding StringPiece for easy access.
  if (shouldTailMerge()) {
    for (MergeInputSection<ELFT> *Sec : Sections) {
      for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I) {
        if (!Sec->Pieces[I].Live)
          continue;
        Sec->Pieces[I].OutputOff = Builder.getOffset(Sec->getData(I));
      }
    }
  }
}

template <class ELFT>
VersionDefinitionSection<ELFT>::VersionDefinitionSection()
    : OutputSectionBase(".gnu.version_d", SHT_GNU_verdef, SHF_ALLOC) {
  this->Addralign = sizeof(uint32_t);
}

static StringRef getFileDefName() {
  if (!Config->SoName.empty())
    return Config->SoName;
  return Config->OutputFile;
}

template <class ELFT> void VersionDefinitionSection<ELFT>::finalize() {
  FileDefNameOff = In<ELFT>::DynStrTab->addString(getFileDefName());
  for (VersionDefinition &V : Config->VersionDefinitions)
    V.NameOff = In<ELFT>::DynStrTab->addString(V.Name);

  this->Size = (sizeof(Elf_Verdef) + sizeof(Elf_Verdaux)) * getVerDefNum();
  this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;

  // sh_info should be set to the number of definitions. This fact is missed in
  // documentation, but confirmed by binutils community:
  // https://sourceware.org/ml/binutils/2014-11/msg00355.html
  this->Info = getVerDefNum();
}

template <class ELFT>
void VersionDefinitionSection<ELFT>::writeOne(uint8_t *Buf, uint32_t Index,
                                              StringRef Name, size_t NameOff) {
  auto *Verdef = reinterpret_cast<Elf_Verdef *>(Buf);
  Verdef->vd_version = 1;
  Verdef->vd_cnt = 1;
  Verdef->vd_aux = sizeof(Elf_Verdef);
  Verdef->vd_next = sizeof(Elf_Verdef) + sizeof(Elf_Verdaux);
  Verdef->vd_flags = (Index == 1 ? VER_FLG_BASE : 0);
  Verdef->vd_ndx = Index;
  Verdef->vd_hash = hashSysV(Name);

  auto *Verdaux = reinterpret_cast<Elf_Verdaux *>(Buf + sizeof(Elf_Verdef));
  Verdaux->vda_name = NameOff;
  Verdaux->vda_next = 0;
}

template <class ELFT>
void VersionDefinitionSection<ELFT>::writeTo(uint8_t *Buf) {
  writeOne(Buf, 1, getFileDefName(), FileDefNameOff);

  for (VersionDefinition &V : Config->VersionDefinitions) {
    Buf += sizeof(Elf_Verdef) + sizeof(Elf_Verdaux);
    writeOne(Buf, V.Id, V.Name, V.NameOff);
  }

  // Need to terminate the last version definition.
  Elf_Verdef *Verdef = reinterpret_cast<Elf_Verdef *>(Buf);
  Verdef->vd_next = 0;
}

template <class ELFT>
VersionTableSection<ELFT>::VersionTableSection()
    : OutputSectionBase(".gnu.version", SHT_GNU_versym, SHF_ALLOC) {
  this->Addralign = sizeof(uint16_t);
}

template <class ELFT> void VersionTableSection<ELFT>::finalize() {
  this->Size =
      sizeof(Elf_Versym) * (In<ELFT>::DynSymTab->getSymbols().size() + 1);
  this->Entsize = sizeof(Elf_Versym);
  // At the moment of june 2016 GNU docs does not mention that sh_link field
  // should be set, but Sun docs do. Also readelf relies on this field.
  this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;
}

template <class ELFT> void VersionTableSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *OutVersym = reinterpret_cast<Elf_Versym *>(Buf) + 1;
  for (const SymbolTableEntry &S : In<ELFT>::DynSymTab->getSymbols()) {
    OutVersym->vs_index = S.Symbol->symbol()->VersionId;
    ++OutVersym;
  }
}

template <class ELFT>
VersionNeedSection<ELFT>::VersionNeedSection()
    : OutputSectionBase(".gnu.version_r", SHT_GNU_verneed, SHF_ALLOC) {
  this->Addralign = sizeof(uint32_t);

  // Identifiers in verneed section start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL.
  // First identifiers are reserved by verdef section if it exist.
  NextIndex = getVerDefNum() + 1;
}

template <class ELFT>
void VersionNeedSection<ELFT>::addSymbol(SharedSymbol<ELFT> *SS) {
  if (!SS->Verdef) {
    SS->symbol()->VersionId = VER_NDX_GLOBAL;
    return;
  }
  SharedFile<ELFT> *F = SS->file();
  // If we don't already know that we need an Elf_Verneed for this DSO, prepare
  // to create one by adding it to our needed list and creating a dynstr entry
  // for the soname.
  if (F->VerdefMap.empty())
    Needed.push_back({F, In<ELFT>::DynStrTab->addString(F->getSoName())});
  typename SharedFile<ELFT>::NeededVer &NV = F->VerdefMap[SS->Verdef];
  // If we don't already know that we need an Elf_Vernaux for this Elf_Verdef,
  // prepare to create one by allocating a version identifier and creating a
  // dynstr entry for the version name.
  if (NV.Index == 0) {
    NV.StrTab = In<ELFT>::DynStrTab->addString(
        SS->file()->getStringTable().data() + SS->Verdef->getAux()->vda_name);
    NV.Index = NextIndex++;
  }
  SS->symbol()->VersionId = NV.Index;
}

template <class ELFT> void VersionNeedSection<ELFT>::writeTo(uint8_t *Buf) {
  // The Elf_Verneeds need to appear first, followed by the Elf_Vernauxs.
  auto *Verneed = reinterpret_cast<Elf_Verneed *>(Buf);
  auto *Vernaux = reinterpret_cast<Elf_Vernaux *>(Verneed + Needed.size());

  for (std::pair<SharedFile<ELFT> *, size_t> &P : Needed) {
    // Create an Elf_Verneed for this DSO.
    Verneed->vn_version = 1;
    Verneed->vn_cnt = P.first->VerdefMap.size();
    Verneed->vn_file = P.second;
    Verneed->vn_aux =
        reinterpret_cast<char *>(Vernaux) - reinterpret_cast<char *>(Verneed);
    Verneed->vn_next = sizeof(Elf_Verneed);
    ++Verneed;

    // Create the Elf_Vernauxs for this Elf_Verneed. The loop iterates over
    // VerdefMap, which will only contain references to needed version
    // definitions. Each Elf_Vernaux is based on the information contained in
    // the Elf_Verdef in the source DSO. This loop iterates over a std::map of
    // pointers, but is deterministic because the pointers refer to Elf_Verdef
    // data structures within a single input file.
    for (auto &NV : P.first->VerdefMap) {
      Vernaux->vna_hash = NV.first->vd_hash;
      Vernaux->vna_flags = 0;
      Vernaux->vna_other = NV.second.Index;
      Vernaux->vna_name = NV.second.StrTab;
      Vernaux->vna_next = sizeof(Elf_Vernaux);
      ++Vernaux;
    }

    Vernaux[-1].vna_next = 0;
  }
  Verneed[-1].vn_next = 0;
}

template <class ELFT> void VersionNeedSection<ELFT>::finalize() {
  this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;
  this->Info = Needed.size();
  unsigned Size = Needed.size() * sizeof(Elf_Verneed);
  for (std::pair<SharedFile<ELFT> *, size_t> &P : Needed)
    Size += P.first->VerdefMap.size() * sizeof(Elf_Vernaux);
  this->Size = Size;
}

template <class ELFT>
static typename ELFT::uint getOutFlags(InputSectionBase<ELFT> *S) {
  return S->Flags & ~SHF_GROUP & ~SHF_COMPRESSED;
}

template <class ELFT>
static SectionKey<ELFT::Is64Bits> createKey(InputSectionBase<ELFT> *C,
                                            StringRef OutsecName) {
  typedef typename ELFT::uint uintX_t;
  uintX_t Flags = getOutFlags(C);

  // For SHF_MERGE we create different output sections for each alignment.
  // This makes each output section simple and keeps a single level mapping from
  // input to output.
  // In case of relocatable object generation we do not try to perform merging
  // and treat SHF_MERGE sections as regular ones, but also create different
  // output sections for them to allow merging at final linking stage.
  uintX_t Alignment = 0;
  if (isa<MergeInputSection<ELFT>>(C) ||
      (Config->Relocatable && (C->Flags & SHF_MERGE)))
    Alignment = std::max<uintX_t>(C->Alignment, C->Entsize);

  return SectionKey<ELFT::Is64Bits>{OutsecName, C->Type, Flags, Alignment};
}

template <class ELFT>
std::pair<OutputSectionBase *, bool>
OutputSectionFactory<ELFT>::create(InputSectionBase<ELFT> *C,
                                   StringRef OutsecName) {
  SectionKey<ELFT::Is64Bits> Key = createKey(C, OutsecName);
  return create(Key, C);
}

template <class ELFT>
std::pair<OutputSectionBase *, bool>
OutputSectionFactory<ELFT>::create(const SectionKey<ELFT::Is64Bits> &Key,
                                   InputSectionBase<ELFT> *C) {
  uintX_t Flags = getOutFlags(C);
  OutputSectionBase *&Sec = Map[Key];
  if (Sec) {
    Sec->Flags |= Flags;
    return {Sec, false};
  }

  uint32_t Type = C->Type;
  switch (C->kind()) {
  case InputSectionBase<ELFT>::Regular:
  case InputSectionBase<ELFT>::Synthetic:
    Sec = make<OutputSection<ELFT>>(Key.Name, Type, Flags);
    break;
  case InputSectionBase<ELFT>::EHFrame:
    return {Out<ELFT>::EhFrame, false};
  case InputSectionBase<ELFT>::Merge:
    Sec = make<MergeOutputSection<ELFT>>(Key.Name, Type, Flags, Key.Alignment);
    break;
  }
  return {Sec, true};
}

template <bool Is64Bits>
typename lld::elf::SectionKey<Is64Bits>
DenseMapInfo<lld::elf::SectionKey<Is64Bits>>::getEmptyKey() {
  return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getEmptyKey(), 0, 0, 0};
}

template <bool Is64Bits>
typename lld::elf::SectionKey<Is64Bits>
DenseMapInfo<lld::elf::SectionKey<Is64Bits>>::getTombstoneKey() {
  return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getTombstoneKey(), 0, 0,
                              0};
}

template <bool Is64Bits>
unsigned
DenseMapInfo<lld::elf::SectionKey<Is64Bits>>::getHashValue(const Key &Val) {
  return hash_combine(Val.Name, Val.Type, Val.Flags, Val.Alignment);
}

template <bool Is64Bits>
bool DenseMapInfo<lld::elf::SectionKey<Is64Bits>>::isEqual(const Key &LHS,
                                                           const Key &RHS) {
  return DenseMapInfo<StringRef>::isEqual(LHS.Name, RHS.Name) &&
         LHS.Type == RHS.Type && LHS.Flags == RHS.Flags &&
         LHS.Alignment == RHS.Alignment;
}

namespace llvm {
template struct DenseMapInfo<SectionKey<true>>;
template struct DenseMapInfo<SectionKey<false>>;
}

namespace lld {
namespace elf {

template void OutputSectionBase::writeHeaderTo<ELF32LE>(ELF32LE::Shdr *Shdr);
template void OutputSectionBase::writeHeaderTo<ELF32BE>(ELF32BE::Shdr *Shdr);
template void OutputSectionBase::writeHeaderTo<ELF64LE>(ELF64LE::Shdr *Shdr);
template void OutputSectionBase::writeHeaderTo<ELF64BE>(ELF64BE::Shdr *Shdr);

template class EhFrameHeader<ELF32LE>;
template class EhFrameHeader<ELF32BE>;
template class EhFrameHeader<ELF64LE>;
template class EhFrameHeader<ELF64BE>;

template class PltSection<ELF32LE>;
template class PltSection<ELF32BE>;
template class PltSection<ELF64LE>;
template class PltSection<ELF64BE>;

template class GnuHashTableSection<ELF32LE>;
template class GnuHashTableSection<ELF32BE>;
template class GnuHashTableSection<ELF64LE>;
template class GnuHashTableSection<ELF64BE>;

template class HashTableSection<ELF32LE>;
template class HashTableSection<ELF32BE>;
template class HashTableSection<ELF64LE>;
template class HashTableSection<ELF64BE>;

template class OutputSection<ELF32LE>;
template class OutputSection<ELF32BE>;
template class OutputSection<ELF64LE>;
template class OutputSection<ELF64BE>;

template class EhOutputSection<ELF32LE>;
template class EhOutputSection<ELF32BE>;
template class EhOutputSection<ELF64LE>;
template class EhOutputSection<ELF64BE>;

template class MergeOutputSection<ELF32LE>;
template class MergeOutputSection<ELF32BE>;
template class MergeOutputSection<ELF64LE>;
template class MergeOutputSection<ELF64BE>;

template class VersionTableSection<ELF32LE>;
template class VersionTableSection<ELF32BE>;
template class VersionTableSection<ELF64LE>;
template class VersionTableSection<ELF64BE>;

template class VersionNeedSection<ELF32LE>;
template class VersionNeedSection<ELF32BE>;
template class VersionNeedSection<ELF64LE>;
template class VersionNeedSection<ELF64BE>;

template class VersionDefinitionSection<ELF32LE>;
template class VersionDefinitionSection<ELF32BE>;
template class VersionDefinitionSection<ELF64LE>;
template class VersionDefinitionSection<ELF64BE>;

template class GdbIndexSection<ELF32LE>;
template class GdbIndexSection<ELF32BE>;
template class GdbIndexSection<ELF64LE>;
template class GdbIndexSection<ELF64BE>;

template class OutputSectionFactory<ELF32LE>;
template class OutputSectionFactory<ELF32BE>;
template class OutputSectionFactory<ELF64LE>;
template class OutputSectionFactory<ELF64BE>;
}
}
