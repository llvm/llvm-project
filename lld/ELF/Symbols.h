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

// Returns a demangled C++ symbol name. If Name is not a mangled
// name or the system does not provide __cxa_demangle function,
// it returns the unmodified string.
std::string demangle(StringRef Name);

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
    SharedKind,
    DefinedElfLast = SharedKind,
    DefinedCommonKind,
    DefinedSyntheticKind,
    DefinedLast = DefinedSyntheticKind,
    UndefinedElfKind,
    UndefinedKind,
    LazyKind
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  bool isWeak() const { return IsWeak; }
  bool isUndefined() const {
    return SymbolKind == UndefinedKind || SymbolKind == UndefinedElfKind;
  }
  bool isDefined() const { return SymbolKind <= DefinedLast; }
  bool isCommon() const { return SymbolKind == DefinedCommonKind; }
  bool isLazy() const { return SymbolKind == LazyKind; }
  bool isShared() const { return SymbolKind == SharedKind; }
  bool isUsedInRegularObj() const { return IsUsedInRegularObj; }
  bool isTls() const { return IsTls; }
  bool isFunc() const { return IsFunc; }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  uint8_t getVisibility() const { return Visibility; }

  unsigned DynsymIndex = 0;
  uint32_t GlobalDynIndex = -1;
  uint32_t GotIndex = -1;
  uint32_t GotPltIndex = -1;
  uint32_t PltIndex = -1;
  bool hasGlobalDynIndex() { return GlobalDynIndex != uint32_t(-1); }
  bool isInGot() const { return GotIndex != -1U; }
  bool isInPlt() const { return PltIndex != -1U; }

  template <class ELFT>
  typename llvm::object::ELFFile<ELFT>::uintX_t getVA() const;
  template <class ELFT>
  typename llvm::object::ELFFile<ELFT>::uintX_t getGotVA() const;
  template <class ELFT>
  typename llvm::object::ELFFile<ELFT>::uintX_t getGotPltVA() const;
  template <class ELFT>
  typename llvm::object::ELFFile<ELFT>::uintX_t getPltVA() const;
  template <class ELFT>
  typename llvm::object::ELFFile<ELFT>::uintX_t getSize() const;

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }
  SymbolBody *repl() { return Backref ? Backref->Body : this; }
  Symbol *getSymbol() { return Backref; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  template <class ELFT> int compare(SymbolBody *Other);

protected:
  SymbolBody(Kind K, StringRef Name, bool IsWeak, uint8_t Visibility,
             bool IsTls, bool IsFunc)
      : SymbolKind(K), IsWeak(IsWeak), Visibility(Visibility),
        MustBeInDynSym(false), IsTls(IsTls), IsFunc(IsFunc), Name(Name) {
    IsUsedInRegularObj = K != SharedKind && K != LazyKind;
  }

  const unsigned SymbolKind : 8;
  unsigned IsWeak : 1;
  unsigned Visibility : 2;

  // True if the symbol was used for linking and thus need to be
  // added to the output file's symbol table. It is usually true,
  // but if it is a shared symbol that were not referenced by anyone,
  // it can be false.
  unsigned IsUsedInRegularObj : 1;

public:
  // If true, the symbol is added to .dynsym symbol table.
  unsigned MustBeInDynSym : 1;

protected:
  unsigned IsTls : 1;
  unsigned IsFunc : 1;
  StringRef Name;
  Symbol *Backref = nullptr;
};

// The base class for any defined symbols.
class Defined : public SymbolBody {
public:
  Defined(Kind K, StringRef Name, bool IsWeak, uint8_t Visibility, bool IsTls,
          bool IsFunction);
  static bool classof(const SymbolBody *S) { return S->isDefined(); }
};

// Any defined symbol from an ELF file.
template <class ELFT> class DefinedElf : public Defined {
protected:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  DefinedElf(Kind K, StringRef N, const Elf_Sym &Sym)
      : Defined(K, N, Sym.getBinding() == llvm::ELF::STB_WEAK,
                Sym.getVisibility(), Sym.getType() == llvm::ELF::STT_TLS,
                Sym.getType() == llvm::ELF::STT_FUNC),
        Sym(Sym) {}

  const Elf_Sym &Sym;
  static bool classof(const SymbolBody *S) {
    return S->kind() <= DefinedElfLast;
  }
};

class DefinedCommon : public Defined {
public:
  DefinedCommon(StringRef N, uint64_t Size, uint64_t Alignment, bool IsWeak,
                uint8_t Visibility);

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedCommonKind;
  }

  // The output offset of this common symbol in the output bss. Computed by the
  // writer.
  uint64_t OffsetInBss;

  // The maximum alignment we have seen for this symbol.
  uint64_t MaxAlignment;

  uint64_t Size;
};

// Regular defined symbols read from object file symbol tables.
template <class ELFT> class DefinedRegular : public DefinedElf<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  DefinedRegular(StringRef N, const Elf_Sym &Sym,
                 InputSectionBase<ELFT> *Section)
      : DefinedElf<ELFT>(SymbolBody::DefinedRegularKind, N, Sym),
        Section(Section) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedRegularKind;
  }

  // If this is null, the symbol is absolute.
  InputSectionBase<ELFT> *Section;
};

// DefinedSynthetic is a class to represent linker-generated ELF symbols.
// The difference from the regular symbol is that DefinedSynthetic symbols
// don't belong to any input files or sections. Thus, its constructor
// takes an output section to calculate output VA, etc.
template <class ELFT> class DefinedSynthetic : public Defined {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  DefinedSynthetic(StringRef N, uintX_t Value,
                   OutputSectionBase<ELFT> &Section);

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::DefinedSyntheticKind;
  }

  uintX_t Value;
  const OutputSectionBase<ELFT> &Section;
};

// Undefined symbol.
class Undefined : public SymbolBody {
  typedef SymbolBody::Kind Kind;
  bool CanKeepUndefined;

protected:
  Undefined(Kind K, StringRef N, bool IsWeak, uint8_t Visibility, bool IsTls);

public:
  Undefined(StringRef N, bool IsWeak, uint8_t Visibility,
            bool CanKeepUndefined);

  static bool classof(const SymbolBody *S) { return S->isUndefined(); }

  bool canKeepUndefined() const { return CanKeepUndefined; }
};

template <class ELFT> class UndefinedElf : public Undefined {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  UndefinedElf(StringRef N, const Elf_Sym &Sym);
  const Elf_Sym &Sym;

  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::UndefinedElfKind;
  }
};

template <class ELFT> class SharedSymbol : public DefinedElf<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  static bool classof(const SymbolBody *S) {
    return S->kind() == SymbolBody::SharedKind;
  }

  SharedSymbol(SharedFile<ELFT> *F, StringRef Name, const Elf_Sym &Sym)
      : DefinedElf<ELFT>(SymbolBody::SharedKind, Name, Sym), File(F) {}

  SharedFile<ELFT> *File;

  // True if the linker has to generate a copy relocation for this shared
  // symbol. OffsetInBss is significant only when NeedsCopy is true.
  bool NeedsCopy = false;
  uintX_t OffsetInBss = 0;
};

// This class represents a symbol defined in an archive file. It is
// created from an archive file header, and it knows how to load an
// object file from an archive to replace itself with a defined
// symbol. If the resolver finds both Undefined and Lazy for
// the same name, it will ask the Lazy to load a file.
class Lazy : public SymbolBody {
public:
  Lazy(ArchiveFile *F, const llvm::object::Archive::Symbol S)
      : SymbolBody(LazyKind, S.getName(), false, llvm::ELF::STV_DEFAULT,
                   /*IsTls*/ false, /*IsFunction*/ false),
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

// Some linker-generated symbols need to be created as
// DefinedRegular symbols, so they need Elf_Sym symbols.
// Here we allocate such Elf_Sym symbols statically.
template <class ELFT> struct ElfSym {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

  // Used to represent an undefined symbol which we don't want to add to the
  // output file's symbol table. It has weak binding and can be substituted.
  static Elf_Sym Ignored;

  // The content for _end and end symbols.
  static Elf_Sym End;

  // The content for _gp symbol for MIPS target.
  static Elf_Sym MipsGp;

  // __rel_iplt_start/__rel_iplt_end for signaling
  // where R_[*]_IRELATIVE relocations do live.
  static Elf_Sym RelaIpltStart;
  static Elf_Sym RelaIpltEnd;
};

template <class ELFT> typename ElfSym<ELFT>::Elf_Sym ElfSym<ELFT>::Ignored;
template <class ELFT> typename ElfSym<ELFT>::Elf_Sym ElfSym<ELFT>::End;
template <class ELFT> typename ElfSym<ELFT>::Elf_Sym ElfSym<ELFT>::MipsGp;
template <class ELFT>
typename ElfSym<ELFT>::Elf_Sym ElfSym<ELFT>::RelaIpltStart;
template <class ELFT> typename ElfSym<ELFT>::Elf_Sym ElfSym<ELFT>::RelaIpltEnd;

} // namespace elf2
} // namespace lld

#endif
