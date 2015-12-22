//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// All symbols are handled as SymbolBodies regardless of their types.
// This file defines various types of SymbolBodies.
//
// File-scope symbols in ELF objects are the only exception of SymbolBody
// instantiation. We will never create SymbolBodies for them for performance
// reason. They are often represented as nullptrs. This is fine for symbol
// resolution because the symbol table naturally cares only about
// externally-visible symbols. For relocations, you have to deal with both
// local and non-local functions, and we have two different functions
// where we need them.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOLS_H
#define LLD_ELF_SYMBOLS_H

#include "InputSection.h"

#include "lld/Core/LLVM.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

class ArchiveFile;
class InputFile;
class SymbolBody;
template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;
template <class ELFT> class SharedFile;

// Initializes global objects defined in this file.
// Called at the beginning of main().
void initSymbols();

// A real symbol object, SymbolBody, is usually accessed indirectly
// through a Symbol. There's always one Symbol for each symbol name.
// The resolver updates SymbolBody pointers as it resolves symbols.
struct Symbol {
  SymbolBody *Body;
};

// The base class for real symbol classes.
class SymbolBody {
public:
  enum Kind {
    DefinedFirst,
    DefinedRegularKind = DefinedFirst,
    DefinedAbsoluteKind,
    DefinedCommonKind,
    DefinedSyntheticKind,
    SharedKind,
    DefinedLast = SharedKind,
    UndefinedKind,
    LazyKind
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  bool isWeak() const { return IsWeak; }
  bool isUndefined() const { return SymbolKind == UndefinedKind; }
  bool isDefined() const { return SymbolKind <= DefinedLast; }
  bool isCommon() const { return SymbolKind == DefinedCommonKind; }
  bool isLazy() const { return SymbolKind == LazyKind; }
  bool isShared() const { return SymbolKind == SharedKind; }
  bool isUsedInRegularObj() const { return IsUsedInRegularObj; }
  bool isUsedInDynamicReloc() const { return IsUsedInDynamicReloc; }
  void setUsedInDynamicReloc() { IsUsedInDynamicReloc = true; }
  bool isTls() const { return IsTls; }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  uint8_t getVisibility() const { return Visibility; }

  unsigned DynamicSymbolTableIndex = 0;
  uint32_t GlobalDynIndex = -1;
  uint32_t GotIndex = -1;
  uint32_t GotPltIndex = -1;
  uint32_t PltIndex = -1;
  bool hasGlobalDynIndex() { return GlobalDynIndex != uint32_t(-1); }
  bool isInGot() const { return GotIndex != -1U; }
  bool isInGotPlt() const { return GotPltIndex != -1U; }
  bool isInPlt() const { return PltIndex != -1U; }

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }
  SymbolBody *repl() { return Backref ? Backref->Body : this; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  template <class ELFT> int compare(SymbolBody *Other);

protected:
  SymbolBody(Kind K, StringRef Name, bool IsWeak, uint8_t Visibility,
             bool IsTls)
      : SymbolKind(K), IsWeak(IsWeak), Visibility(Visibility), IsTls(IsTls),
        Name(Name) {
    IsUsedInRegularObj = K != SharedKind && K != LazyKind;
    IsUsedInDynamicReloc = 0;
  }

  const unsigned SymbolKind : 8;
  unsigned IsWeak : 1;
  unsigned Visibility : 2;
  unsigned IsUsedInRegularObj : 1;
  unsigned IsUsedInDynamicReloc : 1;
  unsigned IsTls : 1;
  StringRef Name;
  Symbol *Backref = nullptr;
};

// This is for symbols created from elf files and not from the command line.
// Since they come from object files, they have a Elf_Sym.
//
// FIXME: Another alternative is to give every symbol an Elf_Sym. To do that
// we have to delay creating the symbol table until the output format is
// known and some of its methods will be templated. We should experiment with
// that once we have a bit more code.
template <class ELFT> class ELFSymbolBody : public SymbolBody {
protected:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  ELFSymbolBody(Kind K, StringRef Name, const Elf_Sym &Sym)
      : SymbolBody(K, Name, Sym.getBinding() == llvm::ELF::STB_WEAK,
                   Sym.getVisibility(), Sym.getType() == llvm::ELF::STT_TLS),
        Sym(Sym) {}

public:
  const Elf_Sym &Sym;

  static bool classof(const SymbolBody *S) {
    Kind K = S->kind();
    return K >= DefinedFirst && K <= UndefinedKind;
  }
};

// The base class for any defined symbols, including absolute symbols, etc.
template <class ELFT> class Defined : public ELFSymbolBody<ELFT> {
protected:
  typedef typename SymbolBody::Kind Kind;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  Defined(Kind K, StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(K, N, Sym) {}

  static bool classof(const SymbolBody *S) { return S->isDefined(); }
};

template <class ELFT> class DefinedAbsolute : public Defined<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  static Elf_Sym IgnoreUndef;

  // The following symbols must be added early to reserve their places
  // in symbol tables. The value of the symbols are set when all sections
  // are finalized and their addresses are determined.

  // The content for _end and end symbols.
  static Elf_Sym End;

  // The content for _gp symbol for MIPS target.
  static Elf_Sym MipsGp;

  // __rel_iplt_start/__rel_iplt_end for signaling
  // where R_[*]_IRELATIVE relocations do live.
  static Elf_Sym RelaIpltStart;
  static Elf_Sym RelaIpltEnd;

  DefinedAbsolute(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(SymbolBody::DefinedAbsoluteKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedAbsoluteKind;
  }
};

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::IgnoreUndef;

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::End;

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::MipsGp;

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::RelaIpltStart;

template <class ELFT>
typename DefinedAbsolute<ELFT>::Elf_Sym DefinedAbsolute<ELFT>::RelaIpltEnd;

template <class ELFT> class DefinedCommon : public Defined<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  DefinedCommon(StringRef N, const Elf_Sym &Sym)
      : Defined<ELFT>(SymbolBody::DefinedCommonKind, N, Sym) {
    MaxAlignment = Sym.st_value;
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedCommonKind;
  }

  // The output offset of this common symbol in the output bss. Computed by the
  // writer.
  uintX_t OffsetInBSS;

  // The maximum alignment we have seen for this symbol.
  uintX_t MaxAlignment;
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public Defined<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  DefinedRegular(StringRef N, const Elf_Sym &Sym,
                 InputSectionBase<ELFT> &Section)
      : Defined<ELFT>(SymbolBody::DefinedRegularKind, N, Sym),
        Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedRegularKind;
  }

  InputSectionBase<ELFT> &Section;
};

// DefinedSynthetic is a class to represent linker-generated ELF symbols.
// The difference from the regular symbol is that DefinedSynthetic symbols
// don't belong to any input files or sections. Thus, its constructor
// takes an output section to calculate output VA, etc.
template <class ELFT> class DefinedSynthetic : public Defined<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  DefinedSynthetic(StringRef N, const Elf_Sym &Sym,
                   OutputSectionBase<ELFT> &Section)
      : Defined<ELFT>(SymbolBody::DefinedSyntheticKind, N, Sym),
        Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedSyntheticKind;
  }

  const OutputSectionBase<ELFT> &Section;
};

// Undefined symbol.
template <class ELFT> class Undefined : public ELFSymbolBody<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  static Elf_Sym Required;
  static Elf_Sym Optional;

  Undefined(StringRef N, const Elf_Sym &Sym)
      : ELFSymbolBody<ELFT>(SymbolBody::UndefinedKind, N, Sym) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::UndefinedKind;
  }

  bool canKeepUndefined() const { return &this->Sym == &Optional; }
};

template <class ELFT>
typename Undefined<ELFT>::Elf_Sym Undefined<ELFT>::Required;
template <class ELFT>
typename Undefined<ELFT>::Elf_Sym Undefined<ELFT>::Optional;

template <class ELFT> class SharedSymbol : public Defined<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::SharedKind;
  }

  SharedSymbol(SharedFile<ELFT> *F, StringRef Name, const Elf_Sym &Sym)
      : Defined<ELFT>(SymbolBody::SharedKind, Name, Sym), File(F) {}

  SharedFile<ELFT> *File;

  // True if the linker has to generate a copy relocation for this shared
  // symbol. OffsetInBSS is significant only when NeedsCopy is true.
  bool NeedsCopy = false;
  uintX_t OffsetInBSS = 0;
};

// This class represents a symbol defined in an archive file. It is
// created from an archive file header, and it knows how to load an
// object file from an archive to replace itself with a defined
// symbol. If the resolver finds both Undefined and Lazy for
// the same name, it will ask the Lazy to load a file.
class Lazy : public SymbolBody {
public:
  Lazy(ArchiveFile *F, const llvm::object::Archive::Symbol S)
      : SymbolBody(LazyKind, S.getName(), false, llvm::ELF::STV_DEFAULT, false),
        File(F), Sym(S) {}

  static bool classof(const SymbolBody *S) { return S->kind() == LazyKind; }

  // Returns an object file for this symbol, or a nullptr if the file
  // was already returned.
  std::unique_ptr<InputFile> getMember();

  void setWeak() { IsWeak = true; }
  void setUsedInRegularObj() { IsUsedInRegularObj = true; }

private:
  ArchiveFile *File;
  const llvm::object::Archive::Symbol Sym;
};

} // namespace elf2
} // namespace lld

#endif
