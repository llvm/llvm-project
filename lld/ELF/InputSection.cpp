//===- InputSection.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "OutputSections.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
InputSectionBase<ELFT>::InputSectionBase(ObjectFile<ELFT> *File,
                                         const Elf_Shdr *Header,
                                         Kind SectionKind)
    : Header(Header), File(File), SectionKind(SectionKind) {}

template <class ELFT> StringRef InputSectionBase<ELFT>::getSectionName() const {
  ErrorOr<StringRef> Name = File->getObj().getSectionName(this->Header);
  error(Name);
  return *Name;
}

template <class ELFT>
ArrayRef<uint8_t> InputSectionBase<ELFT>::getSectionData() const {
  ErrorOr<ArrayRef<uint8_t>> Ret =
      this->File->getObj().getSectionContents(this->Header);
  error(Ret);
  return *Ret;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t
InputSectionBase<ELFT>::getOffset(uintX_t Offset) {
  switch (SectionKind) {
  case Regular:
    return cast<InputSection<ELFT>>(this)->OutSecOff + Offset;
  case EHFrame:
    return cast<EHInputSection<ELFT>>(this)->getOffset(Offset);
  case Merge:
    return cast<MergeInputSection<ELFT>>(this)->getOffset(Offset);
  case MipsReginfo:
    // MIPS .reginfo sections are consumed by the linker,
    // so it should never be copied to output.
    llvm_unreachable("MIPS .reginfo reached writeTo().");
  }
  llvm_unreachable("Invalid section kind");
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t
InputSectionBase<ELFT>::getOffset(const Elf_Sym &Sym) {
  return getOffset(Sym.st_value);
}

// Returns a section that Rel relocation is pointing to.
template <class ELFT>
InputSectionBase<ELFT> *
InputSectionBase<ELFT>::getRelocTarget(const Elf_Rel &Rel) {
  // Global symbol
  uint32_t SymIndex = Rel.getSymbol(Config->Mips64EL);
  if (SymbolBody *B = File->getSymbolBody(SymIndex))
    if (auto *D = dyn_cast<DefinedRegular<ELFT>>(B->repl()))
      return D->Section;
  // Local symbol
  if (const Elf_Sym *Sym = File->getLocalSymbol(SymIndex))
    if (InputSectionBase<ELFT> *Sec = File->getSection(*Sym))
      return Sec;
  return nullptr;
}

template <class ELFT>
InputSectionBase<ELFT> *
InputSectionBase<ELFT>::getRelocTarget(const Elf_Rela &Rel) {
  return getRelocTarget(reinterpret_cast<const Elf_Rel &>(Rel));
}

template <class ELFT>
InputSection<ELFT>::InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header)
    : InputSectionBase<ELFT>(F, Header, Base::Regular) {}

template <class ELFT>
bool InputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == Base::Regular;
}

template <class ELFT>
template <bool isRela>
uint8_t *
InputSectionBase<ELFT>::findMipsPairedReloc(uint8_t *Buf, uint32_t SymIndex,
                                            uint32_t Type,
                                            RelIteratorRange<isRela> Rels) {
  // Some MIPS relocations use addend calculated from addend of the relocation
  // itself and addend of paired relocation. ABI requires to compute such
  // combined addend in case of REL relocation record format only.
  // See p. 4-17 at ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (isRela || Config->EMachine != EM_MIPS)
    return nullptr;
  if (Type == R_MIPS_HI16)
    Type = R_MIPS_LO16;
  else if (Type == R_MIPS_PCHI16)
    Type = R_MIPS_PCLO16;
  else if (Type == R_MICROMIPS_HI16)
    Type = R_MICROMIPS_LO16;
  else
    return nullptr;
  for (const auto &RI : Rels) {
    if (RI.getType(Config->Mips64EL) != Type)
      continue;
    if (RI.getSymbol(Config->Mips64EL) != SymIndex)
      continue;
    uintX_t Offset = getOffset(RI.r_offset);
    if (Offset == (uintX_t)-1)
      return nullptr;
    return Buf + Offset;
  }
  return nullptr;
}

template <class ELFT>
static typename llvm::object::ELFFile<ELFT>::uintX_t
getSymSize(SymbolBody &Body) {
  if (auto *SS = dyn_cast<DefinedElf<ELFT>>(&Body))
    return SS->Sym.st_size;
  return 0;
}

template <class ELFT>
template <bool isRela>
void InputSectionBase<ELFT>::relocate(uint8_t *Buf, uint8_t *BufEnd,
                                      RelIteratorRange<isRela> Rels) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  size_t Num = Rels.end() - Rels.begin();
  for (size_t I = 0; I < Num; ++I) {
    const RelType &RI = *(Rels.begin() + I);
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    uint32_t Type = RI.getType(Config->Mips64EL);
    uintX_t Offset = getOffset(RI.r_offset);
    if (Offset == (uintX_t)-1)
      continue;

    uint8_t *BufLoc = Buf + Offset;
    uintX_t AddrLoc = OutSec->getVA() + Offset;
    auto NextRelocs = llvm::make_range(&RI, Rels.end());

    if (Target->isTlsLocalDynamicReloc(Type) &&
        !Target->isTlsOptimized(Type, nullptr)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getLocalTlsIndexVA() +
                              getAddend<ELFT>(RI));
      continue;
    }

    const Elf_Shdr *SymTab = File->getSymbolTable();
    SymbolBody *Body = nullptr;
    if (SymIndex >= SymTab->sh_info)
      Body = File->getSymbolBody(SymIndex)->repl();

    if (Target->isTlsOptimized(Type, Body)) {
      uintX_t SymVA;
      if (!Body)
        SymVA = getLocalRelTarget(*File, RI, 0);
      else if (Target->relocNeedsGot(Type, *Body))
        SymVA = Out<ELFT>::Got->getEntryAddr(*Body);
      else
        SymVA = getSymVA<ELFT>(*Body);
      // By optimizing TLS relocations, it is sometimes needed to skip
      // relocations that immediately follow TLS relocations. This function
      // knows how many slots we need to skip.
      I += Target->relocateTlsOptimize(BufLoc, BufEnd, Type, AddrLoc, SymVA,
                                       *Body);
      continue;
    }

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    uintX_t A = getAddend<ELFT>(RI);
    if (!Body) {
      uintX_t SymVA = getLocalRelTarget(*File, RI, A);
      // We need to adjust SymVA value in case of R_MIPS_GPREL16/32 relocations
      // because they use the following expression to calculate the relocation's
      // result for local symbol: S + A + GP0 - G.
      if (Config->EMachine == EM_MIPS &&
          (Type == R_MIPS_GPREL16 || Type == R_MIPS_GPREL32))
        SymVA += File->getMipsGp0();
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA, 0,
                          findMipsPairedReloc(Buf, SymIndex, Type, NextRelocs));
      continue;
    }

    if (Target->isTlsGlobalDynamicReloc(Type) &&
        !Target->isTlsOptimized(Type, Body)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getGlobalDynAddr(*Body) +
                              getAddend<ELFT>(RI));
      continue;
    }

    uintX_t SymVA = getSymVA<ELFT>(*Body);
    if (Target->relocNeedsPlt(Type, *Body)) {
      SymVA = Out<ELFT>::Plt->getEntryAddr(*Body);
      Type = Target->getPltRefReloc(Type);
    } else if (Target->relocNeedsGot(Type, *Body)) {
      SymVA = Out<ELFT>::Got->getEntryAddr(*Body);
      if (Body->isTls())
        Type = Target->getTlsGotReloc(Type);
    } else if (!Target->needsCopyRel(Type, *Body) &&
               isa<SharedSymbol<ELFT>>(*Body)) {
      continue;
    } else if (Target->isTlsDynReloc(Type, *Body)) {
      continue;
    } else if (Target->isSizeReloc(Type) && canBePreempted(Body, false)) {
      // A SIZE relocation is supposed to set a symbol size, but if a symbol
      // can be preempted, the size at runtime may be different than link time.
      // If that's the case, we leave the field alone rather than filling it
      // with a possibly incorrect value.
      continue;
    } else if (Config->EMachine == EM_MIPS) {
      if (Type == R_MIPS_HI16 && Body == Config->MipsGpDisp)
        SymVA = getMipsGpAddr<ELFT>() - AddrLoc;
      else if (Type == R_MIPS_LO16 && Body == Config->MipsGpDisp)
        SymVA = getMipsGpAddr<ELFT>() - AddrLoc + 4;
    }
    uintX_t Size = getSymSize<ELFT>(*Body);
    Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA + A, Size + A,
                        findMipsPairedReloc(Buf, SymIndex, Type, NextRelocs));
  }
}

template <class ELFT> void InputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (this->Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = this->getSectionData();
  memcpy(Buf + OutSecOff, Data.data(), Data.size());

  ELFFile<ELFT> &EObj = this->File->getObj();
  uint8_t *BufEnd = Buf + OutSecOff + Data.size();
  // Iterate over all relocation sections that apply to this section.
  for (const Elf_Shdr *RelSec : this->RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      this->relocate(Buf, BufEnd, EObj.relas(RelSec));
    else
      this->relocate(Buf, BufEnd, EObj.rels(RelSec));
  }
}

template <class ELFT>
SplitInputSection<ELFT>::SplitInputSection(
    ObjectFile<ELFT> *File, const Elf_Shdr *Header,
    typename InputSectionBase<ELFT>::Kind SectionKind)
    : InputSectionBase<ELFT>(File, Header, SectionKind) {}

template <class ELFT>
EHInputSection<ELFT>::EHInputSection(ObjectFile<ELFT> *F,
                                     const Elf_Shdr *Header)
    : SplitInputSection<ELFT>(F, Header, InputSectionBase<ELFT>::EHFrame) {
  // Mark .eh_frame sections as live by default because there are
  // usually no relocations that point to .eh_frames. Otherwise,
 // the garbage collector would drop all .eh_frame sections.
  this->Live = true;
}

template <class ELFT>
bool EHInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::EHFrame;
}

template <class ELFT>
typename EHInputSection<ELFT>::uintX_t
EHInputSection<ELFT>::getOffset(uintX_t Offset) {
  // The file crtbeginT.o has relocations pointing to the start of an empty
  // .eh_frame that is known to be the first in the link. It does that to
  // identify the start of the output .eh_frame. Handle this special case.
  if (this->getSectionHdr()->sh_size == 0)
    return Offset;
  std::pair<uintX_t, uintX_t> *I = this->getRangeAndSize(Offset).first;
  uintX_t Base = I->second;
  if (Base == uintX_t(-1))
    return -1; // Not in the output

  uintX_t Addend = Offset - I->first;
  return Base + Addend;
}

template <class ELFT>
MergeInputSection<ELFT>::MergeInputSection(ObjectFile<ELFT> *F,
                                           const Elf_Shdr *Header)
    : SplitInputSection<ELFT>(F, Header, InputSectionBase<ELFT>::Merge) {}

template <class ELFT>
bool MergeInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::Merge;
}

template <class ELFT>
std::pair<std::pair<typename ELFFile<ELFT>::uintX_t,
                    typename ELFFile<ELFT>::uintX_t> *,
          typename ELFFile<ELFT>::uintX_t>
SplitInputSection<ELFT>::getRangeAndSize(uintX_t Offset) {
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  uintX_t Size = Data.size();
  if (Offset >= Size)
    error("Entry is past the end of the section");

  // Find the element this offset points to.
  auto I = std::upper_bound(
      Offsets.begin(), Offsets.end(), Offset,
      [](const uintX_t &A, const std::pair<uintX_t, uintX_t> &B) {
        return A < B.first;
      });
  uintX_t End = I == Offsets.end() ? Data.size() : I->first;
  --I;
  return std::make_pair(&*I, End);
}

template <class ELFT>
typename MergeInputSection<ELFT>::uintX_t
MergeInputSection<ELFT>::getOffset(uintX_t Offset) {
  std::pair<std::pair<uintX_t, uintX_t> *, uintX_t> T =
      this->getRangeAndSize(Offset);
  std::pair<uintX_t, uintX_t> *I = T.first;
  uintX_t End = T.second;
  uintX_t Start = I->first;

  // Compute the Addend and if the Base is cached, return.
  uintX_t Addend = Offset - Start;
  uintX_t &Base = I->second;
  if (Base != uintX_t(-1))
    return Base + Addend;

  // Map the base to the offset in the output section and cache it.
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  StringRef Entry = Data.substr(Start, End - Start);
  Base =
      static_cast<MergeOutputSection<ELFT> *>(this->OutSec)->getOffset(Entry);
  return Base + Addend;
}

template <class ELFT>
MipsReginfoInputSection<ELFT>::MipsReginfoInputSection(ObjectFile<ELFT> *F,
                                                       const Elf_Shdr *Hdr)
    : InputSectionBase<ELFT>(F, Hdr, InputSectionBase<ELFT>::MipsReginfo) {
  // Initialize this->Reginfo.
  ArrayRef<uint8_t> D = this->getSectionData();
  if (D.size() != sizeof(Elf_Mips_RegInfo<ELFT>))
    error("Invalid size of .reginfo section");
  Reginfo = reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(D.data());
}

template <class ELFT>
bool MipsReginfoInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::MipsReginfo;
}

namespace lld {
namespace elf2 {
template class InputSectionBase<object::ELF32LE>;
template class InputSectionBase<object::ELF32BE>;
template class InputSectionBase<object::ELF64LE>;
template class InputSectionBase<object::ELF64BE>;

template class InputSection<object::ELF32LE>;
template class InputSection<object::ELF32BE>;
template class InputSection<object::ELF64LE>;
template class InputSection<object::ELF64BE>;

template class EHInputSection<object::ELF32LE>;
template class EHInputSection<object::ELF32BE>;
template class EHInputSection<object::ELF64LE>;
template class EHInputSection<object::ELF64BE>;

template class MergeInputSection<object::ELF32LE>;
template class MergeInputSection<object::ELF32BE>;
template class MergeInputSection<object::ELF64LE>;
template class MergeInputSection<object::ELF64BE>;

template class MipsReginfoInputSection<object::ELF32LE>;
template class MipsReginfoInputSection<object::ELF32BE>;
template class MipsReginfoInputSection<object::ELF64LE>;
template class MipsReginfoInputSection<object::ELF64BE>;
}
}
