//===--- Types.h - Data structures for used-symbol analysis -------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Find referenced files is mostly a matter of translating:
//    AST Node => declaration => source location => file
//
// clang has types for these (DynTypedNode, Decl, SourceLocation, FileID), but
// there are special cases: macros are not declarations, the concrete file where
// a standard library symbol was defined doesn't matter, etc.
//
// We define some slightly more abstract sum types to handle these cases while
// keeping the API clean. For example, Symbol may be a Decl AST node, a macro,
// or a recognized standard library symbol.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_TYPES_H
#define CLANG_INCLUDE_CLEANER_TYPES_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm
namespace clang {
class Decl;
class FileEntry;
class IdentifierInfo;
namespace include_cleaner {

/// We consider a macro to be a different symbol each time it is defined.
struct Macro {
  IdentifierInfo *Name;
  /// The location of the Name where the macro is defined.
  SourceLocation Definition;

  bool operator==(const Macro &S) const { return Definition == S.Definition; }
};

/// An entity that can be referenced in the code.
struct Symbol {
  enum Kind {
    /// A canonical clang declaration.
    Declaration,
    /// A preprocessor macro, as defined in a specific location.
    Macro,
  };

  Symbol(const Decl &D) : Storage(&D) {}
  Symbol(struct Macro M) : Storage(M) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const Symbol &RHS) const { return Storage == RHS.Storage; }

  const Decl &declaration() const { return *std::get<Declaration>(Storage); }
  struct Macro macro() const { return std::get<Macro>(Storage); }

private:
  // Order must match Kind enum!
  std::variant<const Decl *, struct Macro> Storage;

  Symbol(decltype(Storage) Sentinel) : Storage(std::move(Sentinel)) {}
  friend llvm::DenseMapInfo<Symbol>;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Symbol &);

/// Indicates the relation between the reference and the target.
enum class RefType {
  /// Target is named by the reference, e.g. function call.
  Explicit,
  /// Target isn't spelled, e.g. default constructor call in `Foo f;`
  Implicit,
  /// Target's use can't be proven, e.g. a candidate for an unresolved overload.
  Ambiguous,
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, RefType);

/// Indicates that a piece of code refers to a symbol.
struct SymbolReference {
  /// The point in the code that refers to the symbol.
  SourceLocation RefLocation;
  /// The symbol referred to.
  Symbol Target;
  /// Relation type between the reference location and the target.
  RefType RT;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolReference &);

/// Represents a file that provides some symbol. Might not be includeable, e.g.
/// built-in or main-file itself.
struct Header {
  enum Kind {
    /// A source file parsed by clang. (May also be a <built-in> buffer).
    Physical,
    /// A recognized standard library header, like <string>.
    Standard,
    /// A verbatim header spelling, a string quoted with <> or "" that can be
    /// #included directly.
    Verbatim,
  };

  Header(const FileEntry *FE) : Storage(FE) {}
  Header(tooling::stdlib::Header H) : Storage(H) {}
  Header(StringRef VerbatimSpelling) : Storage(VerbatimSpelling) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const Header &RHS) const { return Storage == RHS.Storage; }

  const FileEntry *physical() const { return std::get<Physical>(Storage); }
  tooling::stdlib::Header standard() const {
    return std::get<Standard>(Storage);
  }
  StringRef verbatim() const { return std::get<Verbatim>(Storage); }

private:
  // Order must match Kind enum!
  std::variant<const FileEntry *, tooling::stdlib::Header, StringRef> Storage;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Header &);

/// A single #include directive written in the main file.
struct Include {
  llvm::StringRef Spelled;             // e.g. vector
  const FileEntry *Resolved = nullptr; // e.g. /path/to/c++/v1/vector
                                       // nullptr if the header was not found
  SourceLocation HashLocation;         // of hash in #include <vector>
  unsigned Line = 0;                   // 1-based line number for #include
  bool Angled = false;                 // True if spelled with <angle> quotes.
  std::string quote() const;           // e.g. <vector>
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Include &);

/// A container for all includes present in a file.
/// Supports efficiently hit-testing Headers against Includes.
class Includes {
public:
  void add(const Include &);

  /// All #includes seen, in the order they appear.
  llvm::ArrayRef<Include> all() const { return All; }

  /// Determine #includes that match a header (that provides a used symbol).
  ///
  /// Matching is based on the type of Header specified:
  ///  - for a physical file like /path/to/foo.h, we check Resolved
  ///  - for a logical file like <vector>, we check Spelled
  llvm::SmallVector<const Include *> match(Header H) const;

  /// Finds the include written on the specified line.
  const Include *atLine(unsigned OneBasedIndex) const;

private:
  std::vector<Include> All;
  // Lookup structures for match(), values are index into All.
  llvm::StringMap<llvm::SmallVector<unsigned>> BySpelling;
  llvm::DenseMap<const FileEntry *, llvm::SmallVector<unsigned>> ByFile;
  llvm::DenseMap<unsigned, unsigned> ByLine;
};

} // namespace include_cleaner
} // namespace clang

namespace llvm {

template <> struct DenseMapInfo<clang::include_cleaner::Symbol> {
  using Outer = clang::include_cleaner::Symbol;
  using Base = DenseMapInfo<decltype(Outer::Storage)>;

  static inline Outer getEmptyKey() { return {Base::getEmptyKey()}; }
  static inline Outer getTombstoneKey() { return {Base::getTombstoneKey()}; }
  static unsigned getHashValue(const Outer &Val) {
    return Base::getHashValue(Val.Storage);
  }
  static bool isEqual(const Outer &LHS, const Outer &RHS) {
    return Base::isEqual(LHS.Storage, RHS.Storage);
  }
};
template <> struct DenseMapInfo<clang::include_cleaner::Macro> {
  using Outer = clang::include_cleaner::Macro;
  using Base = DenseMapInfo<decltype(Outer::Definition)>;

  static inline Outer getEmptyKey() { return {nullptr, Base::getEmptyKey()}; }
  static inline Outer getTombstoneKey() {
    return {nullptr, Base::getTombstoneKey()};
  }
  static unsigned getHashValue(const Outer &Val) {
    return Base::getHashValue(Val.Definition);
  }
  static bool isEqual(const Outer &LHS, const Outer &RHS) {
    return Base::isEqual(LHS.Definition, RHS.Definition);
  }
};
} // namespace llvm

#endif
