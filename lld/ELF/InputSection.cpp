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

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

template <class ELFT>
InputSectionBase<ELFT>::InputSectionBase(elf::ObjectFile<ELFT> *File,
                                         const Elf_Shdr *Header,
                                         Kind SectionKind)
    : Header(Header), File(File), SectionKind(SectionKind), Repl(this) {
  // The garbage collector sets sections' Live bits.
  // If GC is disabled, all sections are considered live by default.
  Live = !Config->GcSections;

  // The ELF spec states that a value of 0 means the section has
  // no alignment constraits.
  Align = std::max<uintX_t>(Header->sh_addralign, 1);
}

template <class ELFT> StringRef InputSectionBase<ELFT>::getSectionName() const {
  return check(File->getObj().getSectionName(this->Header));
}

template <class ELFT>
ArrayRef<uint8_t> InputSectionBase<ELFT>::getSectionData() const {
  return check(this->File->getObj().getSectionContents(this->Header));
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
InputSectionBase<ELFT>::getRelocTarget(const Elf_Rel &Rel) const {
  // Global symbol
  uint32_t SymIndex = Rel.getSymbol(Config->Mips64EL);
  SymbolBody &B = File->getSymbolBody(SymIndex).repl();
  InputSectionBase<ELFT> *S = nullptr;
  if (auto *D = dyn_cast<DefinedRegular<ELFT>>(&B))
    S = D->Section;
  if (S)
    return S->Repl;
  return nullptr;
}

template <class ELFT>
InputSectionBase<ELFT> *
InputSectionBase<ELFT>::getRelocTarget(const Elf_Rela &Rel) const {
  return getRelocTarget(reinterpret_cast<const Elf_Rel &>(Rel));
}

template <class ELFT>
InputSection<ELFT>::InputSection(elf::ObjectFile<ELFT> *F,
                                 const Elf_Shdr *Header)
    : InputSectionBase<ELFT>(F, Header, Base::Regular) {}

template <class ELFT>
bool InputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == Base::Regular;
}

template <class ELFT>
InputSectionBase<ELFT> *InputSection<ELFT>::getRelocatedSection() {
  assert(this->Header->sh_type == SHT_RELA || this->Header->sh_type == SHT_REL);
  ArrayRef<InputSectionBase<ELFT> *> Sections = this->File->getSections();
  return Sections[this->Header->sh_info];
}

// This is used for -r. We can't use memcpy to copy relocations because we need
// to update symbol table offset and section index for each relocation. So we
// copy relocations one by one.
template <class ELFT>
template <bool isRela>
void InputSection<ELFT>::copyRelocations(uint8_t *Buf,
                                         RelIteratorRange<isRela> Rels) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  InputSectionBase<ELFT> *RelocatedSection = getRelocatedSection();

  for (const RelType &Rel : Rels) {
    uint32_t SymIndex = Rel.getSymbol(Config->Mips64EL);
    uint32_t Type = Rel.getType(Config->Mips64EL);
    SymbolBody &Body = this->File->getSymbolBody(SymIndex).repl();

    RelType *P = reinterpret_cast<RelType *>(Buf);
    Buf += sizeof(RelType);

    P->r_offset = RelocatedSection->getOffset(Rel.r_offset);
    P->setSymbolAndType(Body.DynsymIndex, Type, Config->Mips64EL);
  }
}

static uint32_t getMipsPairedRelocType(uint32_t Type) {
  if (Config->EMachine != EM_MIPS)
    return R_MIPS_NONE;
  switch (Type) {
  case R_MIPS_HI16:
    return R_MIPS_LO16;
  case R_MIPS_PCHI16:
    return R_MIPS_PCLO16;
  case R_MICROMIPS_HI16:
    return R_MICROMIPS_LO16;
  default:
    return R_MIPS_NONE;
  }
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
  if (isRela || Type == R_MIPS_NONE)
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
    uintX_t Offset = getOffset(RI.r_offset);
    if (Offset == (uintX_t)-1)
      continue;

    uintX_t A = getAddend<ELFT>(RI);
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    uint32_t Type = RI.getType(Config->Mips64EL);
    uint8_t *BufLoc = Buf + Offset;
    uintX_t AddrLoc = OutSec->getVA() + Offset;
    auto NextRelocs = llvm::make_range(&RI, Rels.end());

    if (Target->pointsToLocalDynamicGotEntry(Type) &&
        !Target->canRelaxTls(Type, nullptr)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getTlsIndexVA() + A);
      continue;
    }

    SymbolBody &Body = File->getSymbolBody(SymIndex).repl();

    if (Target->canRelaxTls(Type, &Body)) {
      uintX_t SymVA;
      if (Target->needsGot(Type, Body))
        SymVA = Body.getGotVA<ELFT>();
      else
        SymVA = Body.getVA<ELFT>();
      // By optimizing TLS relocations, it is sometimes needed to skip
      // relocations that immediately follow TLS relocations. This function
      // knows how many slots we need to skip.
      I += Target->relaxTls(BufLoc, BufEnd, Type, AddrLoc, SymVA, Body);
      continue;
    }

    // PPC64 has a special relocation representing the TOC base pointer
    // that does not have a corresponding symbol.
    if (Config->EMachine == EM_PPC64 && RI.getType(false) == R_PPC64_TOC) {
      uintX_t SymVA = getPPC64TocBase() + A;
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA, 0);
      continue;
    }

    if (Target->isTlsGlobalDynamicRel(Type) &&
        !Target->canRelaxTls(Type, &Body)) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getGlobalDynAddr(Body) + A);
      continue;
    }

    uintX_t SymVA = Body.getVA<ELFT>(A);
    bool CBP = canBePreempted(Body);
    uint8_t *PairedLoc = nullptr;

    if (Target->needsPlt<ELFT>(Type, Body)) {
      SymVA = Body.getPltVA<ELFT>() + A;
    } else if (Target->needsGot(Type, Body)) {
      if (Config->EMachine == EM_MIPS && !CBP) {
        if (Body.isLocal()) {
          // R_MIPS_GOT16 relocation against local symbol requires index of
          // a local GOT entry which contains page address corresponds
          // to sum of the symbol address and addend. The addend in that case
          // is calculated using addends from R_MIPS_GOT16 and paired
          // R_MIPS_LO16 relocations.
          const endianness E = ELFT::TargetEndianness;
          uint8_t *LowLoc =
              findMipsPairedReloc(Buf, SymIndex, R_MIPS_LO16, NextRelocs);
          uint64_t AHL = read32<E>(BufLoc) << 16;
          if (LowLoc)
            AHL += SignExtend64<16>(read32<E>(LowLoc));
          SymVA = Out<ELFT>::Got->getMipsLocalPageAddr(SymVA + AHL);
        } else {
          // For non-local symbols GOT entries should contain their full
          // addresses. But if such symbol cannot be preempted, we do not
          // have to put them into the "global" part of GOT and use dynamic
          // linker to determine their actual addresses. That is why we
          // create GOT entries for them in the "local" part of GOT.
          SymVA = Out<ELFT>::Got->getMipsLocalFullAddr(Body) + A;
        }
      } else {
        SymVA = Body.getGotVA<ELFT>() + A;
      }
      if (Body.IsTls)
        Type = Target->getTlsGotRel(Type);
    } else if (Target->isSizeRel(Type) && CBP) {
      // A SIZE relocation is supposed to set a symbol size, but if a symbol
      // can be preempted, the size at runtime may be different than link time.
      // If that's the case, we leave the field alone rather than filling it
      // with a possibly incorrect value.
      continue;
    } else if (Config->EMachine == EM_MIPS) {
      if (Type == R_MIPS_HI16 && &Body == Config->MipsGpDisp) {
        SymVA = getMipsGpAddr<ELFT>() - AddrLoc + A;
      } else if (Type == R_MIPS_LO16 && &Body == Config->MipsGpDisp) {
        SymVA = getMipsGpAddr<ELFT>() - AddrLoc + 4 + A;
      } else if (&Body == Config->MipsLocalGp) {
        SymVA = getMipsGpAddr<ELFT>() + A;
      } else if (Type == R_MIPS_GPREL16 || Type == R_MIPS_GPREL32) {
        // We need to adjust SymVA value in case of R_MIPS_GPREL16/32
        // relocations because they use the following expression to calculate
        // the relocation's result for local symbol: S + A + GP0 - G.
        SymVA += File->getMipsGp0();
      } else {
        PairedLoc = findMipsPairedReloc(
            Buf, SymIndex, getMipsPairedRelocType(Type), NextRelocs);
      }
    } else if (!Target->needsCopyRel<ELFT>(Type, Body) && CBP) {
      continue;
    }
    uintX_t Size = Body.getSize<ELFT>();
    Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA, Size + A,
                        PairedLoc);
  }
}

template <class ELFT> void InputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (this->Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = this->getSectionData();
  ELFFile<ELFT> &EObj = this->File->getObj();

  // That happens with -r. In that case we need fix the relocation position and
  // target. No relocations are applied.
  if (this->Header->sh_type == SHT_RELA) {
    this->copyRelocations(Buf + OutSecOff, EObj.relas(this->Header));
    return;
  }
  if (this->Header->sh_type == SHT_REL) {
    this->copyRelocations(Buf + OutSecOff, EObj.rels(this->Header));
    return;
  }

  memcpy(Buf + OutSecOff, Data.data(), Data.size());

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
void InputSection<ELFT>::replace(InputSection<ELFT> *Other) {
  this->Align = std::max(this->Align, Other->Align);
  Other->Repl = this->Repl;
  Other->Live = false;
}

template <class ELFT>
SplitInputSection<ELFT>::SplitInputSection(
    elf::ObjectFile<ELFT> *File, const Elf_Shdr *Header,
    typename InputSectionBase<ELFT>::Kind SectionKind)
    : InputSectionBase<ELFT>(File, Header, SectionKind) {}

template <class ELFT>
EHInputSection<ELFT>::EHInputSection(elf::ObjectFile<ELFT> *F,
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
MergeInputSection<ELFT>::MergeInputSection(elf::ObjectFile<ELFT> *F,
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
MipsReginfoInputSection<ELFT>::MipsReginfoInputSection(elf::ObjectFile<ELFT> *F,
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

template class elf::InputSectionBase<ELF32LE>;
template class elf::InputSectionBase<ELF32BE>;
template class elf::InputSectionBase<ELF64LE>;
template class elf::InputSectionBase<ELF64BE>;

template class elf::InputSection<ELF32LE>;
template class elf::InputSection<ELF32BE>;
template class elf::InputSection<ELF64LE>;
template class elf::InputSection<ELF64BE>;

template class elf::EHInputSection<ELF32LE>;
template class elf::EHInputSection<ELF32BE>;
template class elf::EHInputSection<ELF64LE>;
template class elf::EHInputSection<ELF64BE>;

template class elf::MergeInputSection<ELF32LE>;
template class elf::MergeInputSection<ELF32BE>;
template class elf::MergeInputSection<ELF64LE>;
template class elf::MergeInputSection<ELF64BE>;

template class elf::MipsReginfoInputSection<ELF32LE>;
template class elf::MipsReginfoInputSection<ELF32BE>;
template class elf::MipsReginfoInputSection<ELF64LE>;
template class elf::MipsReginfoInputSection<ELF64BE>;
