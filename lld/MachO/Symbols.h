//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYMBOLS_H
#define LLD_MACHO_SYMBOLS_H

#include "InputFiles.h"
#include "InputSection.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MathExtras.h"

namespace lld {
namespace macho {

class InputSection;
class MachHeaderSection;

struct StringRefZ {
  StringRefZ(const char *s) : data(s), size(-1) {}
  StringRefZ(StringRef s) : data(s.data()), size(s.size()) {}

  const char *data;
  const uint32_t size;
};

class Symbol {
public:
  enum Kind {
    DefinedKind,
    UndefinedKind,
    CommonKind,
    DylibKind,
    LazyKind,
  };

  virtual ~Symbol() {}

  Kind kind() const { return static_cast<Kind>(symbolKind); }

  StringRef getName() const {
    if (nameSize == (uint32_t)-1)
      nameSize = strlen(nameData);
    return {nameData, nameSize};
  }

  virtual uint64_t getVA() const { return 0; }

  virtual uint64_t getFileOffset() const {
    llvm_unreachable("attempt to get an offset from a non-defined symbol");
  }

  virtual bool isWeakDef() const { llvm_unreachable("cannot be weak def"); }

  // Only undefined or dylib symbols can be weak references. A weak reference
  // need not be satisfied at runtime, e.g. due to the symbol not being
  // available on a given target platform.
  virtual bool isWeakRef() const { llvm_unreachable("cannot be a weak ref"); }

  virtual bool isTlv() const { llvm_unreachable("cannot be TLV"); }

  // Whether this symbol is in the GOT or TLVPointer sections.
  bool isInGot() const { return gotIndex != UINT32_MAX; }

  // Whether this symbol is in the StubsSection.
  bool isInStubs() const { return stubsIndex != UINT32_MAX; }

  // The index of this symbol in the GOT or the TLVPointer section, depending
  // on whether it is a thread-local. A given symbol cannot be referenced by
  // both these sections at once.
  uint32_t gotIndex = UINT32_MAX;

  uint32_t stubsIndex = UINT32_MAX;

  uint32_t symtabIndex = UINT32_MAX;

  InputFile *getFile() const { return file; }

protected:
  Symbol(Kind k, StringRefZ name, InputFile *file)
      : symbolKind(k), nameData(name.data), nameSize(name.size), file(file) {}

  Kind symbolKind;
  const char *nameData;
  mutable uint32_t nameSize;
  InputFile *file;
};

class Defined : public Symbol {
public:
  Defined(StringRefZ name, InputFile *file, InputSection *isec, uint32_t value,
          bool isWeakDef, bool isExternal, bool isPrivateExtern)
      : Symbol(DefinedKind, name, file), isec(isec), value(value),
        overridesWeakDef(false), privateExtern(isPrivateExtern),
        includeInSymtab(true), weakDef(isWeakDef), external(isExternal) {}

  bool isWeakDef() const override { return weakDef; }
  bool isExternalWeakDef() const {
    return isWeakDef() && isExternal() && !privateExtern;
  }
  bool isTlv() const override {
    return !isAbsolute() && isThreadLocalVariables(isec->flags);
  }

  bool isExternal() const { return external; }
  bool isAbsolute() const { return isec == nullptr; }

  uint64_t getVA() const override;
  uint64_t getFileOffset() const override;

  static bool classof(const Symbol *s) { return s->kind() == DefinedKind; }

  InputFile *file;
  InputSection *isec;
  uint32_t value;

  bool overridesWeakDef : 1;
  // Whether this symbol should appear in the output binary's export trie.
  bool privateExtern : 1;
  // Whether this symbol should appear in the output symbol table.
  bool includeInSymtab : 1;

private:
  const bool weakDef : 1;
  const bool external : 1;
};

// This enum does double-duty: as a symbol property, it indicates whether & how
// a dylib symbol is referenced. As a DylibFile property, it indicates the kind
// of referenced symbols contained within the file. If there are both weak
// and strong references to the same file, we will count the file as
// strongly-referenced.
enum class RefState : uint8_t { Unreferenced = 0, Weak = 1, Strong = 2 };

class Undefined : public Symbol {
public:
  Undefined(StringRefZ name, InputFile *file, RefState refState)
      : Symbol(UndefinedKind, name, file), refState(refState) {
    assert(refState != RefState::Unreferenced);
  }

  bool isWeakRef() const override { return refState == RefState::Weak; }

  static bool classof(const Symbol *s) { return s->kind() == UndefinedKind; }

  RefState refState : 2;
};

// On Unix, it is traditionally allowed to write variable definitions without
// initialization expressions (such as "int foo;") to header files. These are
// called tentative definitions.
//
// Using tentative definitions is usually considered a bad practice; you should
// write only declarations (such as "extern int foo;") to header files.
// Nevertheless, the linker and the compiler have to do something to support
// bad code by allowing duplicate definitions for this particular case.
//
// The compiler creates common symbols when it sees tentative definitions.
// (You can suppress this behavior and let the compiler create a regular
// defined symbol by passing -fno-common. -fno-common is the default in clang
// as of LLVM 11.0.) When linking the final binary, if there are remaining
// common symbols after name resolution is complete, the linker converts them
// to regular defined symbols in a __common section.
class CommonSymbol : public Symbol {
public:
  CommonSymbol(StringRefZ name, InputFile *file, uint64_t size, uint32_t align,
               bool isPrivateExtern)
      : Symbol(CommonKind, name, file), size(size),
        align(align != 1 ? align : llvm::PowerOf2Ceil(size)),
        privateExtern(isPrivateExtern) {
    // TODO: cap maximum alignment
  }

  static bool classof(const Symbol *s) { return s->kind() == CommonKind; }

  const uint64_t size;
  const uint32_t align;
  const bool privateExtern;
};

class DylibSymbol : public Symbol {
public:
  DylibSymbol(DylibFile *file, StringRefZ name, bool isWeakDef,
              RefState refState, bool isTlv)
      : Symbol(DylibKind, name, file), refState(refState), weakDef(isWeakDef),
        tlv(isTlv) {}

  bool isWeakDef() const override { return weakDef; }
  bool isWeakRef() const override { return refState == RefState::Weak; }
  bool isReferenced() const { return refState != RefState::Unreferenced; }
  bool isTlv() const override { return tlv; }
  bool isDynamicLookup() const { return file == nullptr; }
  bool hasStubsHelper() const { return stubsHelperIndex != UINT32_MAX; }

  DylibFile *getFile() const {
    assert(!isDynamicLookup());
    return cast<DylibFile>(file);
  }

  static bool classof(const Symbol *s) { return s->kind() == DylibKind; }

  uint32_t stubsHelperIndex = UINT32_MAX;
  uint32_t lazyBindOffset = UINT32_MAX;

  RefState refState : 2;

private:
  const bool weakDef : 1;
  const bool tlv : 1;
};

class LazySymbol : public Symbol {
public:
  LazySymbol(ArchiveFile *file, const llvm::object::Archive::Symbol &sym)
      : Symbol(LazyKind, sym.getName(), file), sym(sym) {}

  ArchiveFile *getFile() const { return cast<ArchiveFile>(file); }
  void fetchArchiveMember();

  static bool classof(const Symbol *s) { return s->kind() == LazyKind; }

private:
  const llvm::object::Archive::Symbol sym;
};

union SymbolUnion {
  alignas(Defined) char a[sizeof(Defined)];
  alignas(Undefined) char b[sizeof(Undefined)];
  alignas(CommonSymbol) char c[sizeof(CommonSymbol)];
  alignas(DylibSymbol) char d[sizeof(DylibSymbol)];
  alignas(LazySymbol) char e[sizeof(LazySymbol)];
};

template <typename T, typename... ArgT>
T *replaceSymbol(Symbol *s, ArgT &&... arg) {
  static_assert(sizeof(T) <= sizeof(SymbolUnion), "SymbolUnion too small");
  static_assert(alignof(T) <= alignof(SymbolUnion),
                "SymbolUnion not aligned enough");
  assert(static_cast<Symbol *>(static_cast<T *>(nullptr)) == nullptr &&
         "Not a Symbol");

  return new (s) T(std::forward<ArgT>(arg)...);
}

} // namespace macho

std::string toString(const macho::Symbol &);
std::string toMachOString(const llvm::object::Archive::Symbol &);

} // namespace lld

#endif
