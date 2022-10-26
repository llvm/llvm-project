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

namespace clang {
class Decl;
class FileEntry;
namespace include_cleaner {

/// An entity that can be referenced in the code.
struct Symbol {
  Symbol(Decl &D) : Storage(&D) {}
  Symbol(tooling::stdlib::Symbol S) : Storage(S) {}

private:
  // FIXME: Add support for macros.
  std::variant<const Decl *, tooling::stdlib::Symbol> Storage;
};

/// Represents a file that provides some symbol. Might not be includeable, e.g.
/// built-in or main-file itself.
struct Header {
  /// A physical (or logical, in case of a builtin) file.
  Header(const FileEntry *FE) : Storage(FE) {}
  /// A logical file representing a stdlib header.
  Header(tooling::stdlib::Header H) : Storage(H) {}

  bool operator==(const Header &RHS) const { return Storage == RHS.Storage; }

private:
  // FIXME: Handle verbatim spellings.
  std::variant<const FileEntry *, tooling::stdlib::Header> Storage;
};

} // namespace include_cleaner
} // namespace clang

#endif

