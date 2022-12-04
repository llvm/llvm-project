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

#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <variant>

namespace clang {
class SourceLocation;
class Decl;
class FileEntry;
class HeaderSearch;
namespace tooling {
class Replacements;
struct IncludeStyle;
} // namespace tooling
namespace include_cleaner {

/// A UsedSymbolCB is a callback invoked for each symbol reference seen.
///
/// References occur at a particular location, refer to a single symbol, and
/// that symbol may be provided by several headers.
/// FIXME: Provide signals about the providing headers so the caller can filter
/// and rank the results.
using UsedSymbolCB = llvm::function_ref<void(const SymbolReference &SymRef,
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
void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes *PI, const SourceManager &, UsedSymbolCB CB);

struct AnalysisResults {
  std::vector<const Include *> Unused;
  std::vector<std::string> Missing; // Spellings, like "<vector>"
};

/// Determine which headers should be inserted or removed from the main file.
/// This exposes conclusions but not reasons: use lower-level walkUsed for that.
AnalysisResults analyze(llvm::ArrayRef<Decl *> ASTRoots,
                        llvm::ArrayRef<SymbolReference> MacroRefs,
                        const Includes &I, const PragmaIncludes *PI,
                        const SourceManager &SM, HeaderSearch &HS);

/// Removes unused includes and inserts missing ones in the main file.
/// Returns the modified main-file code.
/// The FormatStyle must be C++ or ObjC (to support include ordering).
std::string fixIncludes(const AnalysisResults &Results, llvm::StringRef Code,
                        const format::FormatStyle &IncludeStyle);

} // namespace include_cleaner
} // namespace clang

#endif
