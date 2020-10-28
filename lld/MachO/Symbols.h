//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYMBOLS_H
#define LLD_MACHO_SYMBOLS_H

#include "InputSection.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"
#include "llvm/Object/Archive.h"

namespace lld {
namespace macho {

class InputSection;
class MachHeaderSection;
class DylibFile;
class ArchiveFile;

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
    DylibKind,
    LazyKind,
    DSOHandleKind,
  };

  virtual ~Symbol() {}

  Kind kind() const { return static_cast<Kind>(symbolKind); }

  StringRef getName() const { return {name.data, name.size}; }

  virtual uint64_t getVA() const { return 0; }

  virtual uint64_t getFileOffset() const {
    llvm_unreachable("attempt to get an offset from a non-defined symbol");
  }

  virtual bool isWeakDef() const { llvm_unreachable("cannot be weak"); }

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

protected:
  Symbol(Kind k, StringRefZ name) : symbolKind(k), name(name) {}

  Kind symbolKind;
  StringRefZ name;
};

class Defined : public Symbol {
public:
  Defined(StringRefZ name, InputSection *isec, uint32_t value, bool isWeakDef,
          bool isExternal)
      : Symbol(DefinedKind, name), isec(isec), value(value),
        overridesWeakDef(false), weakDef(isWeakDef), external(isExternal) {}

  bool isWeakDef() const override { return weakDef; }

  bool isTlv() const override { return isThreadLocalVariables(isec->flags); }

  bool isExternal() const { return external; }

  static bool classof(const Symbol *s) { return s->kind() == DefinedKind; }

  uint64_t getVA() const override { return isec->getVA() + value; }

  uint64_t getFileOffset() const override {
    return isec->getFileOffset() + value;
  }

  InputSection *isec;
  uint32_t value;

  bool overridesWeakDef : 1;

private:
  const bool weakDef : 1;
  const bool external : 1;
};

class Undefined : public Symbol {
public:
  Undefined(StringRefZ name) : Symbol(UndefinedKind, name) {}

  static bool classof(const Symbol *s) { return s->kind() == UndefinedKind; }
};

class DylibSymbol : public Symbol {
public:
  DylibSymbol(DylibFile *file, StringRefZ name, bool isWeakDef, bool isTlv)
      : Symbol(DylibKind, name), file(file), weakDef(isWeakDef), tlv(isTlv) {}

  bool isWeakDef() const override { return weakDef; }
  bool isTlv() const override { return tlv; }
  bool hasStubsHelper() const { return stubsHelperIndex != UINT32_MAX; }

  static bool classof(const Symbol *s) { return s->kind() == DylibKind; }

  DylibFile *file;
  uint32_t stubsHelperIndex = UINT32_MAX;
  uint32_t lazyBindOffset = UINT32_MAX;

private:
  const bool weakDef;
  const bool tlv;
};

class LazySymbol : public Symbol {
public:
  LazySymbol(ArchiveFile *file, const llvm::object::Archive::Symbol &sym)
      : Symbol(LazyKind, sym.getName()), file(file), sym(sym) {}

  static bool classof(const Symbol *s) { return s->kind() == LazyKind; }

  void fetchArchiveMember();

private:
  ArchiveFile *file;
  const llvm::object::Archive::Symbol sym;
};

// The Itanium C++ ABI requires dylibs to pass a pointer to __cxa_atexit which
// does e.g. cleanup of static global variables. The ABI document says that the
// pointer can point to any address in one of the dylib's segments, but in
// practice ld64 seems to set it to point to the header, so that's what's
// implemented here.
//
// The ARM C++ ABI uses __dso_handle similarly, but I (int3) have not yet
// tested this on an ARM platform.
//
// DSOHandle effectively functions like a Defined symbol, but it doesn't belong
// to an InputSection.
class DSOHandle : public Symbol {
public:
  DSOHandle(const MachHeaderSection *header)
      : Symbol(DSOHandleKind, name), header(header) {}

  const MachHeaderSection *header;

  uint64_t getVA() const override;

  uint64_t getFileOffset() const override;

  bool isWeakDef() const override { return false; }

  bool isTlv() const override { return false; }

  static constexpr StringRef name = "___dso_handle";

  static bool classof(const Symbol *s) { return s->kind() == DSOHandleKind; }
};

union SymbolUnion {
  alignas(Defined) char a[sizeof(Defined)];
  alignas(Undefined) char b[sizeof(Undefined)];
  alignas(DylibSymbol) char c[sizeof(DylibSymbol)];
  alignas(LazySymbol) char d[sizeof(LazySymbol)];
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
} // namespace lld

#endif
