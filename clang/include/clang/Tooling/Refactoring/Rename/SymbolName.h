//===--- SymbolName.h - Clang refactoring library -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_RENAME_SYMBOLNAME_H
#define LLVM_CLANG_TOOLING_REFACTORING_RENAME_SYMBOLNAME_H

#include "clang/AST/DeclarationName.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class LangOptions;

namespace tooling {

/// A name of a symbol.
///
/// Symbol's name can be composed of multiple strings. For example, Objective-C
/// methods can contain multiple argument labels:
///
/// \code
/// - (void) myMethodNamePiece: (int)x anotherNamePieces:(int)y;
/// //       ^~ string 0 ~~~~~         ^~ string 1 ~~~~~
/// \endcode
class SymbolName {
  llvm::SmallVector<std::string, 1> NamePieces;

public:
  SymbolName();

  /// Create a new \c SymbolName with the specified pieces.
  explicit SymbolName(ArrayRef<StringRef> NamePieces);
  explicit SymbolName(ArrayRef<std::string> NamePieces);

  explicit SymbolName(const DeclarationName &Name);

  /// Creates a \c SymbolName from the given string representation.
  ///
  /// For Objective-C symbol names, this splits a selector into multiple pieces
  /// on `:`. For all other languages the name is used as the symbol name.
  SymbolName(StringRef Name, bool IsObjectiveCSelector);
  SymbolName(StringRef Name, const LangOptions &LangOpts);

  ArrayRef<std::string> getNamePieces() const { return NamePieces; }

  /// If this symbol consists of a single piece return it, otherwise return
  /// `None`.
  ///
  /// Only symbols in Objective-C can consist of multiple pieces, so this
  /// function always returns a value for non-Objective-C symbols.
  std::optional<std::string> getSinglePiece() const;

  /// Returns a human-readable version of this symbol name.
  ///
  /// If the symbol consists of multiple pieces (aka. it is an Objective-C
  /// selector/method name), the pieces are separated by `:`, otherwise just an
  /// identifier name.
  std::string getAsString() const;

  void print(raw_ostream &OS) const;

  bool operator==(const SymbolName &Other) const {
    return NamePieces == Other.NamePieces;
  }
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_RENAME_SYMBOLNAME_H
