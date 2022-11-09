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

#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm
namespace clang {
class Decl;
class FileEntry;
namespace include_cleaner {

/// An entity that can be referenced in the code.
struct Symbol {
  enum Kind {
    /// A canonical clang declaration.
    Declaration,
    /// A recognized symbol from the standard library, like std::string.
    Standard,
  };

  Symbol(const Decl &D) : Storage(&D) {}
  Symbol(tooling::stdlib::Symbol S) : Storage(S) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const Symbol &RHS) const { return Storage == RHS.Storage; }

  tooling::stdlib::Symbol standard() const {
    return std::get<Standard>(Storage);
  }
  const Decl &declaration() const { return *std::get<Declaration>(Storage); }

private:
  // FIXME: Add support for macros.
  // Order must match Kind enum!
  std::variant<const Decl *, tooling::stdlib::Symbol> Storage;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Symbol &);

/// Represents a file that provides some symbol. Might not be includeable, e.g.
/// built-in or main-file itself.
struct Header {
  enum Kind {
    /// A source file parsed by clang. (May also be a <built-in> buffer).
    Physical,
    /// A recognized standard library header, like <string>.
    Standard,
  };

  Header(const FileEntry *FE) : Storage(FE) {}
  Header(tooling::stdlib::Header H) : Storage(H) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const Header &RHS) const { return Storage == RHS.Storage; }

  const FileEntry *physical() const { return std::get<Physical>(Storage); }
  tooling::stdlib::Header standard() const {
    return std::get<Standard>(Storage);
  }

private:
  // FIXME: Handle verbatim spellings.
  // Order must match Kind enum!
  std::variant<const FileEntry *, tooling::stdlib::Header> Storage;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Header &);

} // namespace include_cleaner
} // namespace clang

#endif

