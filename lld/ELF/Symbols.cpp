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
using namespace lld::elf;

template <class ELFT>
static typename ELFT::uint getSymVA(const SymbolBody &Body,
                                    typename ELFT::uint &Addend) {
  typedef typename ELFT::uint uintX_t;

  switch (Body.kind()) {
  case SymbolBody::DefinedSyntheticKind: {
    auto &D = cast<DefinedSynthetic<ELFT>>(Body);
    if (D.Value == DefinedSynthetic<ELFT>::SectionEnd)
      return D.Section.getVA() + D.Section.getSize();
    return D.Section.getVA() + D.Value;
  }
  case SymbolBody::DefinedRegularKind: {
    auto &D = cast<DefinedRegular<ELFT>>(Body);
    InputSectionBase<ELFT> *SC = D.Section;

    // According to the ELF spec reference to a local symbol from outside
    // the group are not allowed. Unfortunately .eh_frame breaks that rule
    // and must be treated specially. For now we just replace the symbol with
    // 0.
    if (SC == &InputSection<ELFT>::Discarded)
      return 0;

    // This is an absolute symbol.
    if (!SC)
      return D.Value;

    uintX_t Offset = D.Value;
    if (D.isSection()) {
      Offset += Addend;
      Addend = 0;
    }
    uintX_t VA = SC->OutSec->getVA() + SC->getOffset(Offset);
    if (D.isTls())
      return VA - Out<ELFT>::TlsPhdr->p_vaddr;
    return VA;
  }
  case SymbolBody::DefinedCommonKind:
    return Out<ELFT>::Bss->getVA() + cast<DefinedCommon>(Body).OffsetInBss;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(Body);
    if (!SS.NeedsCopyOrPltAddr)
      return 0;
    if (SS.isFunc())
      return Body.getPltVA<ELFT>();
    return Out<ELFT>::Bss->getVA() + SS.OffsetInBss;
  }
  case SymbolBody::UndefinedElfKind:
  case SymbolBody::UndefinedBitcodeKind:
    return 0;
  case SymbolBody::LazyArchiveKind:
  case SymbolBody::LazyObjectKind:
    assert(Body.Backref->IsUsedInRegularObj && "lazy symbol reached writer");
    return 0;
  case SymbolBody::DefinedBitcodeKind:
    llvm_unreachable("should have been replaced");
  }
  llvm_unreachable("invalid symbol kind");
}

SymbolBody::SymbolBody(Kind K, uint32_t NameOffset, uint8_t StOther,
                       uint8_t Type)
    : SymbolKind(K), Type(Type), Binding(STB_LOCAL), StOther(StOther),
      NameOffset(NameOffset) {
  init();
}

SymbolBody::SymbolBody(Kind K, StringRef Name, uint8_t Binding, uint8_t StOther,
                       uint8_t Type)
    : SymbolKind(K), Type(Type), Binding(Binding), StOther(StOther),
      Name({Name.data(), Name.size()}) {
  assert(!isLocal());
  init();
}

void SymbolBody::init() {
  CanKeepUndefined = false;
  NeedsCopyOrPltAddr = false;
  CanOmitFromDynSym = false;
}

// Returns true if a symbol can be replaced at load-time by a symbol
// with the same name defined in other ELF executable or DSO.
bool SymbolBody::isPreemptible() const {
  if (isLocal())
    return false;

  // Shared symbols resolve to the definition in the DSO.
  if (isShared())
    return true;

  // That's all that can be preempted in a non-DSO.
  if (!Config->Shared)
    return false;

  // Undefined symbols in DSOs can only be preempted if they are strong.
  // Weak symbols just resolve to zero.
  if (isUndefined())
    return !isWeak();

  // -Bsymbolic means that not even default visibility symbols can be preempted.
  if (Config->Bsymbolic || (Config->BsymbolicFunctions && isFunc()))
    return false;

  // Only default visibility symbols that appear in the dynamic symbol table can
  // be preempted.
  return Backref->Visibility == STV_DEFAULT && Backref->includeInDynsym();
}

template <class ELFT>
typename ELFT::uint SymbolBody::getVA(typename ELFT::uint Addend) const {
  typename ELFT::uint OutVA = getSymVA<ELFT>(*this, Addend);
  return OutVA + Addend;
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotVA() const {
  return Out<ELFT>::Got->getVA() + getGotOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotOffset() const {
  return (Out<ELFT>::Got->getMipsLocalEntriesNum() + GotIndex) *
         sizeof(typename ELFT::uint);
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltVA() const {
  return Out<ELFT>::GotPlt->getVA() + getGotPltOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltOffset() const {
  return GotPltIndex * sizeof(typename ELFT::uint);
}

template <class ELFT> typename ELFT::uint SymbolBody::getPltVA() const {
  return Out<ELFT>::Plt->getVA() + Target->PltZeroSize +
         PltIndex * Target->PltEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getThunkVA() const {
  auto *D = cast<DefinedRegular<ELFT>>(this);
  auto *S = cast<InputSection<ELFT>>(D->Section);
  return S->OutSec->getVA() + S->OutSecOff + S->getThunkOff() +
         ThunkIndex * Target->ThunkSize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getSize() const {
  if (const auto *C = dyn_cast<DefinedCommon>(this))
    return C->Size;
  if (const auto *DR = dyn_cast<DefinedRegular<ELFT>>(this))
    return DR->Size;
  if (const auto *S = dyn_cast<SharedSymbol<ELFT>>(this))
    return S->Sym.st_size;
  if (const auto *U = dyn_cast<UndefinedElf<ELFT>>(this))
    return U->Size;
  return 0;
}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
int SymbolBody::compare(SymbolBody *Other) {
  assert(!isLazy() && !Other->isLazy());
  std::tuple<bool, bool, bool> L(isDefined(), !isShared(), !isWeak());
  std::tuple<bool, bool, bool> R(Other->isDefined(), !Other->isShared(),
                                 !Other->isWeak());

  // Compare the two by symbol type.
  if (L > R)
    return -Other->compare(this);
  if (L != R)
    return -1;
  if (!isDefined() || isShared() || isWeak())
    return 1;

  // If both are equal in terms of symbol type, then at least
  // one of them must be a common symbol. Otherwise, they conflict.
  auto *A = dyn_cast<DefinedCommon>(this);
  auto *B = dyn_cast<DefinedCommon>(Other);
  if (!A && !B)
    return 0;

  // If both are common, the larger one is chosen.
  if (A && B) {
    if (Config->WarnCommon)
      warning("multiple common of " + A->getName());
    A->Alignment = B->Alignment = std::max(A->Alignment, B->Alignment);
    return A->Size < B->Size ? -1 : 1;
  }

  // Non-common symbols takes precedence over common symbols.
  if (Config->WarnCommon)
    warning("common " + this->getName() + " is overridden");
  return A ? -1 : 1;
}

Defined::Defined(Kind K, StringRef Name, uint8_t Binding, uint8_t StOther,
                 uint8_t Type)
    : SymbolBody(K, Name, Binding, StOther, Type) {}

Defined::Defined(Kind K, uint32_t NameOffset, uint8_t StOther, uint8_t Type)
    : SymbolBody(K, NameOffset, StOther, Type) {}

DefinedBitcode::DefinedBitcode(StringRef Name, bool IsWeak, uint8_t StOther)
    : Defined(DefinedBitcodeKind, Name, IsWeak ? STB_WEAK : STB_GLOBAL,
              StOther, 0 /* Type */) {}

bool DefinedBitcode::classof(const SymbolBody *S) {
  return S->kind() == DefinedBitcodeKind;
}

UndefinedBitcode::UndefinedBitcode(StringRef N, bool IsWeak, uint8_t StOther)
    : SymbolBody(SymbolBody::UndefinedBitcodeKind, N,
                 IsWeak ? STB_WEAK : STB_GLOBAL, StOther, 0 /* Type */) {}

template <typename ELFT>
UndefinedElf<ELFT>::UndefinedElf(StringRef N, const Elf_Sym &Sym)
    : SymbolBody(SymbolBody::UndefinedElfKind, N, Sym.getBinding(),
                 Sym.st_other, Sym.getType()),
      Size(Sym.st_size) {}

template <typename ELFT>
UndefinedElf<ELFT>::UndefinedElf(StringRef Name, uint8_t Binding,
                                 uint8_t StOther, uint8_t Type,
                                 bool CanKeepUndefined)
    : SymbolBody(SymbolBody::UndefinedElfKind, Name, Binding, StOther, Type) {
  this->CanKeepUndefined = CanKeepUndefined;
}

template <typename ELFT>
UndefinedElf<ELFT>::UndefinedElf(const Elf_Sym &Sym)
    : SymbolBody(SymbolBody::UndefinedElfKind, Sym.st_name, Sym.st_other,
                 Sym.getType()),
      Size(Sym.st_size) {
  assert(Sym.getBinding() == STB_LOCAL);
}

template <typename ELFT>
DefinedSynthetic<ELFT>::DefinedSynthetic(StringRef N, uintX_t Value,
                                         OutputSectionBase<ELFT> &Section)
    : Defined(SymbolBody::DefinedSyntheticKind, N, STB_GLOBAL, STV_HIDDEN,
              0 /* Type */),
      Value(Value), Section(Section) {}

DefinedCommon::DefinedCommon(StringRef N, uint64_t Size, uint64_t Alignment,
                             uint8_t Binding, uint8_t StOther, uint8_t Type)
    : Defined(SymbolBody::DefinedCommonKind, N, Binding, StOther, Type),
      Alignment(Alignment), Size(Size) {}

std::unique_ptr<InputFile> Lazy::getFile() {
  if (auto *S = dyn_cast<LazyArchive>(this))
    return S->getFile();
  return cast<LazyObject>(this)->getFile();
}

std::unique_ptr<InputFile> LazyArchive::getFile() {
  MemoryBufferRef MBRef = File->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBRef.getBuffer().empty())
    return std::unique_ptr<InputFile>(nullptr);
  return createObjectFile(MBRef, File->getName());
}

std::unique_ptr<InputFile> LazyObject::getFile() {
  return createObjectFile(MBRef);
}

// Returns the demangled C++ symbol name for Name.
std::string elf::demangle(StringRef Name) {
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

bool Symbol::includeInDynsym() const {
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return false;
  return (ExportDynamic && VersionScriptGlobal) || Body->isShared();
}

template uint32_t SymbolBody::template getVA<ELF32LE>(uint32_t) const;
template uint32_t SymbolBody::template getVA<ELF32BE>(uint32_t) const;
template uint64_t SymbolBody::template getVA<ELF64LE>(uint64_t) const;
template uint64_t SymbolBody::template getVA<ELF64BE>(uint64_t) const;

template uint32_t SymbolBody::template getGotVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotOffset<ELF32LE>() const;
template uint32_t SymbolBody::template getGotOffset<ELF32BE>() const;
template uint64_t SymbolBody::template getGotOffset<ELF64LE>() const;
template uint64_t SymbolBody::template getGotOffset<ELF64BE>() const;

template uint32_t SymbolBody::template getGotPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotPltOffset<ELF32LE>() const;
template uint32_t SymbolBody::template getGotPltOffset<ELF32BE>() const;
template uint64_t SymbolBody::template getGotPltOffset<ELF64LE>() const;
template uint64_t SymbolBody::template getGotPltOffset<ELF64BE>() const;

template uint32_t SymbolBody::template getPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64BE>() const;

template uint32_t SymbolBody::template getSize<ELF32LE>() const;
template uint32_t SymbolBody::template getSize<ELF32BE>() const;
template uint64_t SymbolBody::template getSize<ELF64LE>() const;
template uint64_t SymbolBody::template getSize<ELF64BE>() const;

template uint32_t SymbolBody::template getThunkVA<ELF32LE>() const;
template uint32_t SymbolBody::template getThunkVA<ELF32BE>() const;
template uint64_t SymbolBody::template getThunkVA<ELF64LE>() const;
template uint64_t SymbolBody::template getThunkVA<ELF64BE>() const;

template class elf::UndefinedElf<ELF32LE>;
template class elf::UndefinedElf<ELF32BE>;
template class elf::UndefinedElf<ELF64LE>;
template class elf::UndefinedElf<ELF64BE>;

template class elf::DefinedSynthetic<ELF32LE>;
template class elf::DefinedSynthetic<ELF32BE>;
template class elf::DefinedSynthetic<ELF64LE>;
template class elf::DefinedSynthetic<ELF64BE>;
