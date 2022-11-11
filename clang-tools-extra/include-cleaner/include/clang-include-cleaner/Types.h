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
    /// A recognized symbol from the standard library, like std::string.
    Standard,
  };

  Symbol(const Decl &D) : Storage(&D) {}
  Symbol(struct Macro M) : Storage(M) {}
  Symbol(tooling::stdlib::Symbol S) : Storage(S) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const Symbol &RHS) const { return Storage == RHS.Storage; }

  const Decl &declaration() const { return *std::get<Declaration>(Storage); }
  struct Macro macro() const { return std::get<Macro>(Storage); }
  tooling::stdlib::Symbol standard() const {
    return std::get<Standard>(Storage);
  }

private:
  // FIXME: Add support for macros.
  // Order must match Kind enum!
  std::variant<const Decl *, struct Macro, tooling::stdlib::Symbol> Storage;
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
  StringRef verbatim() const {
    return std::get<Verbatim>(Storage);
  }

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
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Include &);

} // namespace include_cleaner
} // namespace clang

#endif
