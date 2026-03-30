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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace clang {
class SourceLocation;
class FileID;
class SourceManager;
class Decl;
class FileEntry;
class HeaderSearch;
namespace tooling {
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
/// It only reports references from main file.
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
              const PragmaIncludes *PI, const Preprocessor &PP,
              UsedSymbolCB CB);
/// Overload that allows customizing which FileIDs are treated as "main file".
/// The predicate is evaluated on the expansion file of a reference.
void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes *PI, const Preprocessor &PP, UsedSymbolCB CB,
              llvm::function_ref<bool(FileID)> IsMainFile);

/// A location kind where a symbol reference is observed.
enum class SymbolReferenceOrigin {
  MainFile,
  Preamble,
  Fragment,
};

/// A missing include finding with per-reference provenance.
struct MissingIncludeRef {
  SymbolReference Ref;
  llvm::SmallVector<Header> Providers;
  SymbolReferenceOrigin Origin = SymbolReferenceOrigin::MainFile;
  // Null if the fragment file has multiple direct include sites.
  const Include *FragmentInclude = nullptr;
};

/// An include kept alive only by fragment usage.
struct FragmentDependency {
  const Include *Preserved = nullptr;
  llvm::SmallVector<const Include *> Fragments;
};

/// The result of include-cleaner analysis for one main file.
struct AnalysisResults {
  std::vector<const Include *> Unused;
  // Deduplicated insertion plan, e.g. "<vector>" paired with the chosen
  // provider Header.
  std::vector<std::pair<std::string, Header>> MissingIncludes;
  // Per-reference provenance for consumers that need richer diagnostics.
  std::vector<MissingIncludeRef> MissingRefs;
  std::vector<FragmentDependency> FragmentDependencies;
};

/// Analysis configuration shared by include-cleaner consumers.
struct AnalysisOptions {
  /// No analysis will be performed for headers that satisfy the predicate.
  std::function<bool(const Header &)> HeaderFilter;
  /// A predicate matched against normalized resolved paths, and normalized
  /// spelled paths as a fallback, to identify direct include fragments.
  std::function<bool(llvm::StringRef)> FragmentHeaderFilter;
};

/// Determine which headers should be inserted or removed from the main file.
/// This exposes conclusions but not reasons: use lower-level walkUsed for that.
AnalysisResults analyze(llvm::ArrayRef<Decl *> ASTRoots,
                        llvm::ArrayRef<SymbolReference> MacroRefs,
                        const Includes &I, const PragmaIncludes *PI,
                        const Preprocessor &PP,
                        const AnalysisOptions &Options = {});

enum class FragmentDependencyCommentStatus {
  CanInsert,
  AlreadyPresent,
  ConflictingComment,
};

/// Planned comment state for an include kept alive by fragments.
struct FragmentDependencyComment {
  const Include *Preserved = nullptr;
  llvm::SmallVector<const Include *> Fragments;
  std::string Text;
  FragmentDependencyCommentStatus Status =
      FragmentDependencyCommentStatus::CanInsert;
  std::optional<tooling::Replacement> Replacement;
};

/// Replacements computed from include-cleaner findings.
struct IncludeFixes {
  tooling::Replacements Replacements;
  std::vector<FragmentDependencyComment> FragmentComments;
};

/// Options for turning analysis results into source edits.
struct FixIncludesOptions {
  /// Raw trailing comment text without the leading //.
  ///
  /// When it contains `{0}`, that placeholder is replaced with a comma-
  /// separated list of direct fragment include spellings that keep the include
  /// alive.
  llvm::StringRef FragmentDependencyCommentFormat;
};

/// Computes replacements to apply include-cleaner findings to the main file.
IncludeFixes computeIncludeFixes(const AnalysisResults &Results,
                                 llvm::StringRef FileName, llvm::StringRef Code,
                                 const format::FormatStyle &IncludeStyle,
                                 const FixIncludesOptions &Options = {});

/// Removes unused includes and inserts missing ones in the main file.
/// Returns the modified main-file code.
/// The FormatStyle must be C++ or ObjC (to support include ordering).
std::string fixIncludes(const AnalysisResults &Results,
                        llvm::StringRef FileName, llvm::StringRef Code,
                        const format::FormatStyle &IncludeStyle);
std::string fixIncludes(const AnalysisResults &Results,
                        llvm::StringRef FileName, llvm::StringRef Code,
                        const format::FormatStyle &IncludeStyle,
                        const FixIncludesOptions &Options);

/// Gets all the providers for a symbol by traversing each location.
/// Returned headers are sorted by relevance, first element is the most
/// likely provider for the symbol.
llvm::SmallVector<Header> headersForSymbol(const Symbol &S,
                                           const Preprocessor &PP,
                                           const PragmaIncludes *PI);
} // namespace include_cleaner
} // namespace clang

#endif
