//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Error.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"

#ifdef HAVE_CXXABI_H
#include <cxxabi.h>
#endif

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
typename ELFFile<ELFT>::uintX_t SymbolBody::getVA() const {
  switch (kind()) {
  case DefinedSyntheticKind: {
    auto *D = cast<DefinedSynthetic<ELFT>>(this);
    return D->Section.getVA() + D->Value;
  }
  case DefinedRegularKind: {
    auto *D = cast<DefinedRegular<ELFT>>(this);
    InputSectionBase<ELFT> *SC = D->Section;

    // This is an absolute symbol.
    if (!SC)
      return D->Sym.st_value;

    // Symbol offsets for AMDGPU need to be the offset in bytes of the symbol
    // from the beginning of the section.
    if (Config->EMachine == EM_AMDGPU)
      return SC->getOffset(D->Sym);
    if (D->Sym.getType() == STT_TLS)
      return SC->OutSec->getVA() + SC->getOffset(D->Sym) -
             Out<ELFT>::TlsPhdr->p_vaddr;
    return SC->OutSec->getVA() + SC->getOffset(D->Sym);
  }
  case DefinedCommonKind:
    return Out<ELFT>::Bss->getVA() + cast<DefinedCommon>(this)->OffsetInBss;
  case SharedKind: {
    auto *SS = cast<SharedSymbol<ELFT>>(this);
    if (SS->NeedsCopy)
      return Out<ELFT>::Bss->getVA() + SS->OffsetInBss;
    return 0;
  }
  case UndefinedElfKind:
  case UndefinedKind:
    return 0;
  case LazyKind:
    assert(isUsedInRegularObj() && "Lazy symbol reached writer");
    return 0;
  }
  llvm_unreachable("Invalid symbol kind");
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t SymbolBody::getGotVA() const {
  return Out<ELFT>::Got->getVA() +
         (Out<ELFT>::Got->getMipsLocalEntriesNum() + GotIndex) *
             sizeof(typename ELFFile<ELFT>::uintX_t);
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t SymbolBody::getGotPltVA() const {
  return Out<ELFT>::GotPlt->getVA() +
         GotPltIndex * sizeof(typename ELFFile<ELFT>::uintX_t);
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t SymbolBody::getPltVA() const {
  return Out<ELFT>::Plt->getVA() + Target->PltZeroSize +
         PltIndex * Target->PltEntrySize;
}

static uint8_t getMinVisibility(uint8_t VA, uint8_t VB) {
  if (VA == STV_DEFAULT)
    return VB;
  if (VB == STV_DEFAULT)
    return VA;
  return std::min(VA, VB);
}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
template <class ELFT> int SymbolBody::compare(SymbolBody *Other) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  assert(!isLazy() && !Other->isLazy());
  std::tuple<bool, bool, bool> L(isDefined(), !isShared(), !isWeak());
  std::tuple<bool, bool, bool> R(Other->isDefined(), !Other->isShared(),
                                 !Other->isWeak());

  // Normalize
  if (L > R)
    return -Other->compare<ELFT>(this);

  Visibility = Other->Visibility =
      getMinVisibility(Visibility, Other->Visibility);

  if (IsUsedInRegularObj || Other->IsUsedInRegularObj)
    IsUsedInRegularObj = Other->IsUsedInRegularObj = true;

  // We want to export all symbols that exist both in the executable
  // and in DSOs, so that the symbols in the executable can interrupt
  // symbols in the DSO at runtime.
  if (isShared() != Other->isShared())
    if (isa<DefinedRegular<ELFT>>(isShared() ? Other : this))
      IsUsedInDynamicReloc = Other->IsUsedInDynamicReloc = true;

  if (L != R)
    return -1;
  if (!std::get<0>(L) || !std::get<1>(L) || !std::get<2>(L))
    return 1;
  if (isCommon()) {
    if (!Other->isCommon())
      return -1;
    auto *ThisC = cast<DefinedCommon>(this);
    auto *OtherC = cast<DefinedCommon>(Other);
    uintX_t Align = std::max(ThisC->MaxAlignment, OtherC->MaxAlignment);
    if (ThisC->Size >= OtherC->Size) {
      ThisC->MaxAlignment = Align;
      return 1;
    }
    OtherC->MaxAlignment = Align;
    return -1;
  }
  if (Other->isCommon())
    return 1;
  return 0;
}

Defined::Defined(Kind K, StringRef Name, bool IsWeak, uint8_t Visibility,
                 bool IsTls)
    : SymbolBody(K, Name, IsWeak, Visibility, IsTls) {}

Undefined::Undefined(SymbolBody::Kind K, StringRef N, bool IsWeak,
                     uint8_t Visibility, bool IsTls)
    : SymbolBody(K, N, IsWeak, Visibility, IsTls), CanKeepUndefined(false) {}

Undefined::Undefined(StringRef N, bool IsWeak, uint8_t Visibility,
                     bool CanKeepUndefined)
    : Undefined(SymbolBody::UndefinedKind, N, IsWeak, Visibility,
                /*IsTls*/ false) {
  this->CanKeepUndefined = CanKeepUndefined;
}

template <typename ELFT>
UndefinedElf<ELFT>::UndefinedElf(StringRef N, const Elf_Sym &Sym)
    : Undefined(SymbolBody::UndefinedElfKind, N,
                Sym.getBinding() == llvm::ELF::STB_WEAK, Sym.getVisibility(),
                Sym.getType() == llvm::ELF::STT_TLS),
      Sym(Sym) {}

template <typename ELFT>
DefinedSynthetic<ELFT>::DefinedSynthetic(StringRef N, uintX_t Value,
                                         OutputSectionBase<ELFT> &Section)
    : Defined(SymbolBody::DefinedSyntheticKind, N, false, STV_DEFAULT, false),
      Value(Value), Section(Section) {}

DefinedCommon::DefinedCommon(StringRef N, uint64_t Size, uint64_t Alignment,
                             bool IsWeak, uint8_t Visibility)
    : Defined(SymbolBody::DefinedCommonKind, N, IsWeak, Visibility, false) {
  MaxAlignment = Alignment;
  this->Size = Size;
}

std::unique_ptr<InputFile> Lazy::getMember() {
  MemoryBufferRef MBRef = File->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBRef.getBuffer().empty())
    return std::unique_ptr<InputFile>(nullptr);
  return createObjectFile(MBRef);
}

template <class ELFT> static void doInitSymbols() {
  ElfSym<ELFT>::End.setBinding(STB_GLOBAL);
  ElfSym<ELFT>::Ignored.setBinding(STB_WEAK);
  ElfSym<ELFT>::Ignored.setVisibility(STV_HIDDEN);
}

void elf2::initSymbols() {
  doInitSymbols<ELF32LE>();
  doInitSymbols<ELF32BE>();
  doInitSymbols<ELF64LE>();
  doInitSymbols<ELF64BE>();
}

// Returns the demangled C++ symbol name for Name.
std::string elf2::demangle(StringRef Name) {
#if !defined(HAVE_CXXABI_H)
  return Name;
#else
  if (!Config->Demangle)
    return Name;

  // __cxa_demangle can be used to demangle strings other than symbol
  // names which do not necessarily start with "_Z". Name can be
  // either a C or C++ symbol. Don't call __cxa_demangle if the name
  // does not look like a C++ symbol name to avoid getting unexpected
  // result for a C symbol that happens to match a mangled type name.
  if (!Name.startswith("_Z"))
    return Name;

  char *Buf =
      abi::__cxa_demangle(Name.str().c_str(), nullptr, nullptr, nullptr);
  if (!Buf)
    return Name;
  std::string S(Buf);
  free(Buf);
  return S;
#endif
}

template uint32_t SymbolBody::template getVA<ELF32LE>() const;
template uint32_t SymbolBody::template getVA<ELF32BE>() const;
template uint64_t SymbolBody::template getVA<ELF64LE>() const;
template uint64_t SymbolBody::template getVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64BE>() const;

template uint32_t SymbolBody::template getPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64BE>() const;

template int SymbolBody::compare<ELF32LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF32BE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64BE>(SymbolBody *Other);

template class elf2::UndefinedElf<ELF32LE>;
template class elf2::UndefinedElf<ELF32BE>;
template class elf2::UndefinedElf<ELF64LE>;
template class elf2::UndefinedElf<ELF64BE>;

template class elf2::DefinedSynthetic<ELF32LE>;
template class elf2::DefinedSynthetic<ELF32BE>;
template class elf2::DefinedSynthetic<ELF64LE>;
template class elf2::DefinedSynthetic<ELF64BE>;
