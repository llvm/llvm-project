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
#include "LinkerScript.h"
#include "Memory.h"
#include "Strings.h"
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

template <typename ELFT>
static bool compareByFilePosition(InputSection<ELFT> *A,
                                  InputSection<ELFT> *B) {
  // Synthetic doesn't have link order dependecy, stable_sort will keep it last
  if (A->kind() == InputSectionData::Synthetic ||
      B->kind() == InputSectionData::Synthetic)
    return false;
  auto *LA = cast<InputSection<ELFT>>(A->getLinkOrderDep());
  auto *LB = cast<InputSection<ELFT>>(B->getLinkOrderDep());
  OutputSectionBase *AOut = LA->OutSec;
  OutputSectionBase *BOut = LB->OutSec;
  if (AOut != BOut)
    return AOut->SectionIndex < BOut->SectionIndex;
  return LA->OutSecOff < LB->OutSecOff;
}

template <class ELFT> void OutputSection<ELFT>::finalize() {
  if ((this->Flags & SHF_LINK_ORDER) && !this->Sections.empty()) {
    std::sort(Sections.begin(), Sections.end(), compareByFilePosition<ELFT>);
    Size = 0;
    assignOffsets();

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

// Fill [Buf, Buf + Size) with Filler. Filler is written in big
// endian order. This is used for linker script "=fillexp" command.
void fill(uint8_t *Buf, size_t Size, uint32_t Filler) {
  uint8_t V[4];
  write32be(V, Filler);
  size_t I = 0;
  for (; I + 4 < Size; I += 4)
    memcpy(Buf + I, V, 4);
  memcpy(Buf + I, V, Size - I);
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (uint32_t Filler = Script<ELFT>::X->getFiller(this->Name))
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
                                         ArrayRef<RelTy> Rels) {
  auto *Sec = cast<EhInputSection<ELFT>>(Piece.ID);
  const endianness E = ELFT::TargetEndianness;
  if (read32<E>(Piece.data().data() + 4) != 0)
    fatal(toString(Sec) + ": CIE expected at beginning of .eh_frame");

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
                                      ArrayRef<RelTy> Rels) {
  auto *Sec = cast<EhInputSection<ELFT>>(Piece.ID);
  unsigned FirstRelI = Piece.FirstRelocation;
  if (FirstRelI == (unsigned)-1)
    fatal(toString(Sec) + ": FDE doesn't reference another section");
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
      OffsetToCie[Offset] = addCie(Piece, Rels);
      continue;
    }

    uint32_t CieOffset = Offset + 4 - ID;
    CieRecord *Cie = OffsetToCie[CieOffset];
    if (!Cie)
      fatal(toString(Sec) + ": invalid CIE reference");

    if (!isFdeLive(Piece, Rels))
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
  if (In<ELFT>::EhFrameHdr) {
    for (CieRecord *Cie : Cies) {
      uint8_t Enc = getFdeEncoding<ELFT>(Cie->Piece);
      for (SectionPiece *Fde : Cie->FdePieces) {
        uintX_t Pc = getFdePc(Buf, Fde->OutputOff, Enc);
        uintX_t FdeVA = this->Addr + Fde->OutputOff;
        In<ELFT>::EhFrameHdr->addFde(Pc, FdeVA);
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
  if (shouldTailMerge())
    for (MergeInputSection<ELFT> *Sec : Sections)
      for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I)
        if (Sec->Pieces[I].Live)
          Sec->Pieces[I].OutputOff = Builder.getOffset(Sec->getData(I));
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

template class OutputSectionFactory<ELF32LE>;
template class OutputSectionFactory<ELF32BE>;
template class OutputSectionFactory<ELF64LE>;
template class OutputSectionFactory<ELF64BE>;
}
}
