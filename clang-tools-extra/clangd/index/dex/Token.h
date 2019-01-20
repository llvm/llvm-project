//===--- Token.h - Symbol Search primitive ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Token objects represent a characteristic of a symbol, which can be used to
/// perform efficient search. Tokens are keys for inverted index which are
/// mapped to the corresponding posting lists.
///
/// The symbol std::cout might have the tokens:
/// * Scope "std::"
/// * Trigram "cou"
/// * Trigram "out"
/// * Type "std::ostream"
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TOKEN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TOKEN_H

#include "index/Index.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

/// A Token represents an attribute of a symbol, such as a particular trigram
/// present in the name (used for fuzzy search).
///
/// Tokens can be used to perform more sophisticated search queries by
/// constructing complex iterator trees.
struct Token {
  /// Kind specifies Token type which defines semantics for the internal
  /// representation. Each Kind has different representation stored in Data
  /// field.
  // FIXME(kbobyrev): Storing Data hash would be more efficient than storing raw
  // strings. For example, PathURI store URIs of each directory and its parents,
  // which induces a lot of overhead because these paths tend to be long and
  // each parent directory is a prefix.
  enum class Kind {
    /// Represents trigram used for fuzzy search of unqualified symbol names.
    ///
    /// Data contains 3 bytes with trigram contents.
    Trigram,
    /// Scope primitives, e.g. "symbol belongs to namespace foo::bar".
    ///
    /// Data stroes full scope name, e.g. "foo::bar::baz::" or "" (for global
    /// scope).
    Scope,
    /// Path Proximity URI to symbol declaration.
    ///
    /// Data stores path URI of symbol declaration file or its parent.
    ///
    /// Example: "file:///path/to/clang-tools-extra/clangd/index/SymbolIndex.h"
    /// and some amount of its parents.
    ProximityURI,
    /// Internal Token type for invalid/special tokens, e.g. empty tokens for
    /// llvm::DenseMap.
    Sentinel,
    /// FIXME(kbobyrev): Add other Token Kinds
    /// * Type with qualified type name or its USR
  };

  Token(Kind TokenKind, llvm::StringRef Data)
      : Data(Data), TokenKind(TokenKind) {}

  bool operator==(const Token &Other) const {
    return TokenKind == Other.TokenKind && Data == Other.Data;
  }

  /// Representation which is unique among Token with the same Kind.
  std::string Data;
  Kind TokenKind;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Token &T) {
    switch (T.TokenKind) {
    case Kind::Trigram:
      OS << "T=";
      break;
    case Kind::Scope:
      OS << "S=";
      break;
    case Kind::ProximityURI:
      OS << "U=";
      break;
    case Kind::Sentinel:
      OS << "?=";
      break;
    }
    return OS << T.Data;
  }

private:
  friend llvm::hash_code hash_value(const Token &Token) {
    return llvm::hash_combine(static_cast<int>(Token.TokenKind), Token.Data);
  }
};

} // namespace dex
} // namespace clangd
} // namespace clang

namespace llvm {

// Support Tokens as DenseMap keys.
template <> struct DenseMapInfo<clang::clangd::dex::Token> {
  static inline clang::clangd::dex::Token getEmptyKey() {
    return {clang::clangd::dex::Token::Kind::Sentinel, "EmptyKey"};
  }

  static inline clang::clangd::dex::Token getTombstoneKey() {
    return {clang::clangd::dex::Token::Kind::Sentinel, "TombstoneKey"};
  }

  static unsigned getHashValue(const clang::clangd::dex::Token &Tag) {
    return hash_value(Tag);
  }

  static bool isEqual(const clang::clangd::dex::Token &LHS,
                      const clang::clangd::dex::Token &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TOKEN_H
