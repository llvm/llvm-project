//===--- Analysis.h - Analyze symbol references in AST ------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// A library that provides usage analysis for symbols based on AST analysis.
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_ANALYSIS_H
#define CLANG_INCLUDE_CLEANER_ANALYSIS_H

#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <variant>

namespace clang {
class SourceLocation;
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
/// A UsedSymbolCB is a callback invoked for each symbol reference seen.
///
/// References occur at a particular location, refer to a single symbol, and
/// that symbol may be provided by several headers.
/// FIXME: Provide signals about the reference type and providing headers so the
/// caller can filter and rank the results.
using UsedSymbolCB = llvm::function_ref<void(
    SourceLocation RefLoc, Symbol Target, llvm::ArrayRef<Header> Providers)>;

/// Find and report all references to symbols in a region of code.
///
/// The AST traversal is rooted at ASTRoots - typically top-level declarations
/// of a single source file.
/// FIXME: Handle macro uses.
///
/// This is the main entrypoint of the include-cleaner library, and can be used:
///  - to diagnose missing includes: a referenced symbol is provided by
///    headers which don't match any #include in the main file
///  - to diagnose unused includes: an #include in the main file does not match
///    the headers for any referenced symbol
/// FIXME: Take in an include structure to improve location to header mappings
/// (e.g. IWYU pragmas).
void walkUsed(llvm::ArrayRef<Decl *> ASTRoots, UsedSymbolCB CB);

} // namespace include_cleaner
} // namespace clang

#endif
