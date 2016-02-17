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
  fatal(Name);
  return *Name;
}

template <class ELFT>
ArrayRef<uint8_t> InputSectionBase<ELFT>::getSectionData() const {
  ErrorOr<ArrayRef<uint8_t>> Ret =
      this->File->getObj().getSectionContents(this->Header);
  fatal(Ret);
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

    if (Target->isTlsLocalDynamicRel(Type) &&
        !Target->canRelaxTls(Type, nullptr)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getTlsIndexVA() +
                              getAddend<ELFT>(RI));
      continue;
    }

    const Elf_Shdr *SymTab = File->getSymbolTable();
    SymbolBody *Body = nullptr;
    if (SymIndex >= SymTab->sh_info)
      Body = File->getSymbolBody(SymIndex)->repl();

    if (Target->canRelaxTls(Type, Body)) {
      uintX_t SymVA;
      if (!Body)
        SymVA = getLocalRelTarget(*File, RI, 0);
      else if (Target->needsGot(Type, *Body))
        SymVA = Body->getGotVA<ELFT>();
      else
        SymVA = Body->getVA<ELFT>();
      // By optimizing TLS relocations, it is sometimes needed to skip
      // relocations that immediately follow TLS relocations. This function
      // knows how many slots we need to skip.
      I += Target->relaxTls(BufLoc, BufEnd, Type, AddrLoc, SymVA, Body);
      continue;
    }

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    uintX_t A = getAddend<ELFT>(RI);
    if (!Body) {
      uintX_t SymVA = getLocalRelTarget(*File, RI, A);
      if (Config->EMachine == EM_MIPS) {
        if (Type == R_MIPS_GPREL16 || Type == R_MIPS_GPREL32)
          // We need to adjust SymVA value in case of R_MIPS_GPREL16/32
          // relocations because they use the following expression to calculate
          // the relocation's result for local symbol: S + A + GP0 - G.
          SymVA += File->getMipsGp0();
        else if (Type == R_MIPS_GOT16)
          // R_MIPS_GOT16 relocation against local symbol requires index of
          // a local GOT entry which contains page address corresponds
          // to the symbol address.
          SymVA = Out<ELFT>::Got->getMipsLocalPageAddr(SymVA);
      }
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA, 0,
                          findMipsPairedReloc(Buf, SymIndex, Type, NextRelocs));
      continue;
    }

    if (Target->isTlsGlobalDynamicRel(Type) &&
        !Target->canRelaxTls(Type, Body)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getGlobalDynAddr(*Body) +
                              getAddend<ELFT>(RI));
      continue;
    }

    uintX_t SymVA = Body->getVA<ELFT>();
    if (Target->needsPlt(Type, *Body)) {
      SymVA = Body->getPltVA<ELFT>();
    } else if (Target->needsGot(Type, *Body)) {
      if (Config->EMachine == EM_MIPS && !canBePreempted(Body, true))
        // Under some conditions relocations against non-local symbols require
        // entries in the local part of MIPS GOT. In that case we need an entry
        // initialized by full address of the symbol.
        SymVA = Out<ELFT>::Got->getMipsLocalFullAddr(*Body);
      else
        SymVA = Body->getGotVA<ELFT>();
      if (Body->isTls())
        Type = Target->getTlsGotRel(Type);
    } else if (!Target->needsCopyRel(Type, *Body) &&
               isa<SharedSymbol<ELFT>>(*Body)) {
      continue;
    } else if (Target->isTlsDynRel(Type, *Body)) {
      continue;
    } else if (Target->isSizeRel(Type) && canBePreempted(Body, false)) {
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
      else if (Body == Config->MipsLocalGp)
        SymVA = getMipsGpAddr<ELFT>();
    }
    uintX_t Size = Body->getSize<ELFT>();
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
    fatal("Entry is past the end of the section");

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
    fatal("Invalid size of .reginfo section");
  Reginfo = reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(D.data());
}

template <class ELFT>
bool MipsReginfoInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::MipsReginfo;
}

template class elf2::InputSectionBase<ELF32LE>;
template class elf2::InputSectionBase<ELF32BE>;
template class elf2::InputSectionBase<ELF64LE>;
template class elf2::InputSectionBase<ELF64BE>;

template class elf2::InputSection<ELF32LE>;
template class elf2::InputSection<ELF32BE>;
template class elf2::InputSection<ELF64LE>;
template class elf2::InputSection<ELF64BE>;

template class elf2::EHInputSection<ELF32LE>;
template class elf2::EHInputSection<ELF32BE>;
template class elf2::EHInputSection<ELF64LE>;
template class elf2::EHInputSection<ELF64BE>;

template class elf2::MergeInputSection<ELF32LE>;
template class elf2::MergeInputSection<ELF32BE>;
template class elf2::MergeInputSection<ELF64LE>;
template class elf2::MergeInputSection<ELF64BE>;

template class elf2::MipsReginfoInputSection<ELF32LE>;
template class elf2::MipsReginfoInputSection<ELF32BE>;
template class elf2::MipsReginfoInputSection<ELF64LE>;
template class elf2::MipsReginfoInputSection<ELF64BE>;
