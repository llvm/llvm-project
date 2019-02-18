//===--- SymbolID.h ----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLID_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLID_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <string>

namespace clang {
namespace clangd {

// The class identifies a particular C++ symbol (class, function, method, etc).
//
// As USRs (Unified Symbol Resolution) could be large, especially for functions
// with long type arguments, SymbolID is using truncated SHA1(USR) values to
// guarantee the uniqueness of symbols while using a relatively small amount of
// memory (vs storing USRs directly).
//
// SymbolID can be used as key in the symbol indexes to lookup the symbol.
class SymbolID {
public:
  SymbolID() = default;
  explicit SymbolID(llvm::StringRef USR);

  bool operator==(const SymbolID &Sym) const {
    return HashValue == Sym.HashValue;
  }
  bool operator<(const SymbolID &Sym) const {
    return HashValue < Sym.HashValue;
  }

  // The stored hash is truncated to RawSize bytes.
  // This trades off memory against the number of symbols we can handle.
  constexpr static size_t RawSize = 8;
  llvm::StringRef raw() const;
  static SymbolID fromRaw(llvm::StringRef);

  // Returns a hex encoded string.
  std::string str() const;
  static llvm::Expected<SymbolID> fromStr(llvm::StringRef);

private:
  std::array<uint8_t, RawSize> HashValue;
};

llvm::hash_code hash_value(const SymbolID &ID);

// Write SymbolID into the given stream. SymbolID is encoded as ID.str().
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolID &ID);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLID_H
