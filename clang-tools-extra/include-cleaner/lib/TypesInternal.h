//===--- TypesInternal.h - Intermediate structures used for analysis C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_TYPESINTERNAL_H
#define CLANG_INCLUDE_CLEANER_TYPESINTERNAL_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/BitmaskEnum.h"
#include <cstdint>
#include <utility>
#include <variant>

namespace llvm {
class raw_ostream;
}
namespace clang::include_cleaner {
/// A place where a symbol can be provided.
/// It is either a physical file of the TU (SourceLocation) or a logical
/// location in the standard library (stdlib::Symbol).
struct SymbolLocation {
  enum Kind {
    /// A position within a source file (or macro expansion) parsed by clang.
    Physical,
    /// A recognized standard library symbol, like std::string.
    Standard,
  };

  SymbolLocation(SourceLocation S) : Storage(S) {}
  SymbolLocation(tooling::stdlib::Symbol S) : Storage(S) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const SymbolLocation &RHS) const {
    return Storage == RHS.Storage;
  }
  SourceLocation physical() const { return std::get<Physical>(Storage); }
  tooling::stdlib::Symbol standard() const {
    return std::get<Standard>(Storage);
  }

private:
  // Order must match Kind enum!
  std::variant<SourceLocation, tooling::stdlib::Symbol> Storage;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolLocation &);

/// Represents properties of a symbol provider.
///
/// Hints represents the properties of the edges traversed when finding headers
/// that satisfy an AST node (AST node => symbols => locations => headers).
///
/// Since there can be multiple paths from an AST node to same header, we need
/// to merge hints. These hints are merged by taking the union of all the
/// properties along all the paths. We choose the boolean sense accordingly,
/// e.g. "Public" rather than "Private", because a header is good if it provides
/// any public definition, even if it also provides private ones.
///
/// Hints are sorted in ascending order of relevance.
enum class Hints : uint8_t {
  None = 0x00,
  /// Symbol is directly originating from this header, rather than being
  /// exported or included transitively.
  OriginHeader = 1 << 0,
  /// Provides a generally-usable definition for the symbol. (a function decl,
  /// or class definition and not a forward declaration of a template).
  CompleteSymbol = 1 << 1,
  /// Symbol is provided by a public file. Only absent in the cases where file
  /// is explicitly marked as such, non self-contained or IWYU private
  /// pragmas.
  PublicHeader = 1 << 2,
  /// Header providing the symbol is explicitly marked as preferred, with an
  /// IWYU private pragma that points at this provider or header and symbol has
  /// ~the same name.
  PreferredHeader = 1 << 3,
  LLVM_MARK_AS_BITMASK_ENUM(PreferredHeader),
};
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
/// A wrapper to augment values with hints.
template <typename T> struct Hinted : public T {
  Hints Hint;
  Hinted(T &&Wrapped, Hints H) : T(std::move(Wrapped)), Hint(H) {}

  /// Since hints are sorted by relevance, use it directly.
  bool operator<(const Hinted<T> &Other) const {
    return static_cast<int>(Hint) < static_cast<int>(Other.Hint);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const Hinted<T> &H) {
    return OS << static_cast<int>(H.Hint) << " - " << static_cast<T>(H);
  }
};

} // namespace clang::include_cleaner

#endif
