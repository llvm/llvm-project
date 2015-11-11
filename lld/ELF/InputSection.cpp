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
InputSectionBase<ELFT>::getOffset(const Elf_Sym &Sym) {
  if (auto *S = dyn_cast<InputSection<ELFT>>(this))
    return S->OutSecOff + Sym.st_value;
  return cast<MergeInputSection<ELFT>>(this)->getOffset(Sym.st_value);
}

// Returns a section that Rel relocation is pointing to.
template <class ELFT>
InputSectionBase<ELFT> *
InputSectionBase<ELFT>::getRelocTarget(const Elf_Rel &Rel) {
  // Global symbol
  uint32_t SymIndex = Rel.getSymbol(Config->Mips64EL);
  if (SymbolBody *B = File->getSymbolBody(SymIndex))
    if (auto *D = dyn_cast<DefinedRegular<ELFT>>(B->repl()))
      return &D->Section;
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
void InputSection<ELFT>::relocate(
    uint8_t *Buf, uint8_t *BufEnd,
    iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
    const ObjectFile<ELFT> &File, uintX_t BaseAddr) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    uint32_t Type = RI.getType(Config->Mips64EL);
    uint8_t *BufLoc = Buf + RI.r_offset;
    uintX_t AddrLoc = BaseAddr + RI.r_offset;

    if (Type == Target->getTlsLocalDynamicReloc()) {
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                          Out<ELFT>::Got->getVA() +
                              Out<ELFT>::LocalModuleTlsIndexOffset +
                              getAddend<ELFT>(RI));
      continue;
    }

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    const Elf_Shdr *SymTab = File.getSymbolTable();
    if (SymIndex < SymTab->sh_info) {
      uintX_t SymVA = getLocalRelTarget(File, RI);
      Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc, SymVA);
      continue;
    }

    SymbolBody &Body = *File.getSymbolBody(SymIndex)->repl();
    uintX_t SymVA = getSymVA<ELFT>(Body);
    if (Target->relocNeedsPlt(Type, Body)) {
      SymVA = Out<ELFT>::Plt->getEntryAddr(Body);
      Type = Target->getPLTRefReloc(Type);
    } else if (Target->relocNeedsGot(Type, Body)) {
      SymVA = Out<ELFT>::Got->getEntryAddr(Body);
      Type = Target->getGotRefReloc();
    } else if (Target->relocPointsToGot(Type)) {
      SymVA = Out<ELFT>::Got->getVA();
      Type = Target->getPCRelReloc();
    } else if (!Target->relocNeedsCopy(Type, Body) &&
               isa<SharedSymbol<ELFT>>(Body)) {
      continue;
    }
    Target->relocateOne(BufLoc, BufEnd, Type, AddrLoc,
                        SymVA + getAddend<ELFT>(RI));
  }
}

template <class ELFT> void InputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (this->Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = this->getSectionData();
  memcpy(Buf + OutSecOff, Data.data(), Data.size());

  ELFFile<ELFT> &EObj = this->File->getObj();
  uint8_t *Base = Buf + OutSecOff;
  uintX_t BaseAddr = this->OutSec->getVA() + OutSecOff;
  // Iterate over all relocation sections that apply to this section.
  for (const Elf_Shdr *RelSec : RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      relocate(Base, Base + Data.size(), EObj.relas(RelSec), *this->File,
               BaseAddr);
    else
      relocate(Base, Base + Data.size(), EObj.rels(RelSec), *this->File,
               BaseAddr);
  }
}

template <class ELFT>
MergeInputSection<ELFT>::MergeInputSection(ObjectFile<ELFT> *F,
                                           const Elf_Shdr *Header)
    : InputSectionBase<ELFT>(F, Header, Base::Merge) {}

template <class ELFT>
bool MergeInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == Base::Merge;
}

template <class ELFT>
typename MergeInputSection<ELFT>::uintX_t
MergeInputSection<ELFT>::getOffset(uintX_t Offset) {
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  uintX_t Size = Data.size();
  if (Offset >= Size)
    error("Entry is past the end of the section");

  // Find the element this offset points to.
  auto I = std::upper_bound(
      this->Offsets.begin(), this->Offsets.end(), Offset,
      [](const uintX_t &A, const std::pair<uintX_t, size_t> &B) {
        return A < B.first;
      });
  size_t End = I == this->Offsets.end() ? Data.size() : I->first;
  --I;
  uintX_t Start = I->first;

  // Compute the Addend and if the Base is cached, return.
  uintX_t Addend = Offset - Start;
  size_t &Base = I->second;
  if (Base != size_t(-1))
    return Base + Addend;

  // Map the base to the offset in the output section and cashe it.
  StringRef Entry = Data.substr(Start, End - Start);
  Base =
      static_cast<MergeOutputSection<ELFT> *>(this->OutSec)->getOffset(Entry);
  return Base + Addend;
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

template class MergeInputSection<object::ELF32LE>;
template class MergeInputSection<object::ELF32BE>;
template class MergeInputSection<object::ELF64LE>;
template class MergeInputSection<object::ELF64BE>;
}
}
