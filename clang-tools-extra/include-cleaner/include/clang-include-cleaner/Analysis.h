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

#include "clang-include-cleaner/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <variant>

namespace clang {
class SourceLocation;
class Decl;
class FileEntry;
namespace include_cleaner {

/// A UsedSymbolCB is a callback invoked for each symbol reference seen.
///
/// References occur at a particular location, refer to a single symbol, and
/// that symbol may be provided by several headers.
/// FIXME: Provide signals about the providing headers so the caller can filter
/// and rank the results.
using UsedSymbolCB = llvm::function_ref<void(SymbolReference SymRef,
                                             llvm::ArrayRef<Header> Providers)>;

/// Find and report all references to symbols in a region of code.
///
/// The AST traversal is rooted at ASTRoots - typically top-level declarations
/// of a single source file.
/// The references to macros must be recorded separately and provided.
///
/// This is the main entrypoint of the include-cleaner library, and can be used:
///  - to diagnose missing includes: a referenced symbol is provided by
///    headers which don't match any #include in the main file
///  - to diagnose unused includes: an #include in the main file does not match
///    the headers for any referenced symbol
/// FIXME: Take in an include structure to improve location to header mappings
/// (e.g. IWYU pragmas).
void walkUsed(const SourceManager &, llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs, UsedSymbolCB CB);

} // namespace include_cleaner
} // namespace clang

#endif
