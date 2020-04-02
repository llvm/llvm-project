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
#include "lld/Common/Strings.h"
#include "llvm/Object/Archive.h"

namespace lld {
namespace macho {

class InputSection;
class InputFile;
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
  };

  Kind kind() const { return static_cast<Kind>(symbolKind); }

  StringRef getName() const { return {name.data, name.size}; }

  uint64_t getVA() const;

  InputFile *file;

protected:
  Symbol(Kind k, InputFile *file, StringRefZ name)
      : file(file), symbolKind(k), name(name) {}

  Kind symbolKind;
  StringRefZ name;
};

class Defined : public Symbol {
public:
  Defined(StringRefZ name, InputSection *isec, uint32_t value)
      : Symbol(DefinedKind, nullptr, name), isec(isec), value(value) {}

  InputSection *isec;
  uint32_t value;

  static bool classof(const Symbol *s) { return s->kind() == DefinedKind; }
};

class Undefined : public Symbol {
public:
  Undefined(StringRefZ name) : Symbol(UndefinedKind, nullptr, name) {}

  static bool classof(const Symbol *s) { return s->kind() == UndefinedKind; }
};

inline uint64_t Symbol::getVA() const {
  if (auto *d = dyn_cast<Defined>(this))
    return d->isec->addr + d->value - ImageBase;
  return 0;
}

union SymbolUnion {
  alignas(Defined) char a[sizeof(Defined)];
  alignas(Undefined) char b[sizeof(Undefined)];
};

template <typename T, typename... ArgT>
void replaceSymbol(Symbol *s, ArgT &&... arg) {
  static_assert(sizeof(T) <= sizeof(SymbolUnion), "SymbolUnion too small");
  static_assert(alignof(T) <= alignof(SymbolUnion),
                "SymbolUnion not aligned enough");
  assert(static_cast<Symbol *>(static_cast<T *>(nullptr)) == nullptr &&
         "Not a Symbol");

  new (s) T(std::forward<ArgT>(arg)...);
}

} // namespace macho

std::string toString(const macho::Symbol &);
} // namespace lld

#endif
