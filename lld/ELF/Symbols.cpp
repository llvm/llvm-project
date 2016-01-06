//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "InputSection.h"
#include "Error.h"
#include "InputFiles.h"

#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

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
  std::pair<bool, bool> L(isDefined(), !isWeak());
  std::pair<bool, bool> R(Other->isDefined(), !Other->isWeak());

  // Normalize
  if (L > R)
    return -Other->compare<ELFT>(this);

  Visibility = Other->Visibility =
      getMinVisibility(Visibility, Other->Visibility);

  if (IsUsedInRegularObj || Other->IsUsedInRegularObj)
    IsUsedInRegularObj = Other->IsUsedInRegularObj = true;

  if (L != R)
    return -1;
  if (!L.first || !L.second)
    return 1;
  if (isShared())
    return -1;
  if (Other->isShared())
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
  ElfSym<ELFT>::IgnoreUndef.setBinding(STB_WEAK);
  ElfSym<ELFT>::IgnoreUndef.setVisibility(STV_HIDDEN);
}

void lld::elf2::initSymbols() {
  doInitSymbols<ELF32LE>();
  doInitSymbols<ELF32BE>();
  doInitSymbols<ELF64LE>();
  doInitSymbols<ELF64BE>();
}

template int SymbolBody::compare<ELF32LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF32BE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64BE>(SymbolBody *Other);

template class lld::elf2::UndefinedElf<ELF32LE>;
template class lld::elf2::UndefinedElf<ELF32BE>;
template class lld::elf2::UndefinedElf<ELF64LE>;
template class lld::elf2::UndefinedElf<ELF64BE>;

template class lld::elf2::DefinedSynthetic<ELF32LE>;
template class lld::elf2::DefinedSynthetic<ELF32BE>;
template class lld::elf2::DefinedSynthetic<ELF64LE>;
template class lld::elf2::DefinedSynthetic<ELF64BE>;
