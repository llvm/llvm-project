//===--- IncludeCleaner.h - Unused/Missing Headers Analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Include Cleaner is clangd functionality for providing diagnostics for misuse
/// of transitive headers and unused includes. It is inspired by
/// Include-What-You-Use tool (https://include-what-you-use.org/). Our goal is
/// to provide useful warnings in most popular scenarios but not 1:1 exact
/// feature compatibility.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDECLEANER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDECLEANER_H

#include "Headers.h"
#include "ParsedAST.h"
#include "index/CanonicalIncludes.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringSet.h"
#include <optional>
#include <vector>

namespace clang {
namespace clangd {

struct ReferencedLocations {
  llvm::DenseSet<SourceLocation> User;
  llvm::DenseSet<tooling::stdlib::Symbol> Stdlib;
};

/// Finds locations of all symbols used in the main file.
///
/// - RecursiveASTVisitor finds references to symbols and records their
///   associated locations. These may be macro expansions, and are not resolved
///   to their spelling or expansion location. These locations are later used to
///   determine which headers should be marked as "used" and "directly used".
/// - If \p Tokens is not nullptr, we also examine all identifier tokens in the
///   file in case they reference macros macros.
/// We use this to compute unused headers, so we:
///
/// - cover the whole file in a single traversal for efficiency
/// - don't attempt to describe where symbols were referenced from in
///   ambiguous cases (e.g. implicitly used symbols, multiple declarations)
/// - err on the side of reporting all possible locations
ReferencedLocations findReferencedLocations(ASTContext &Ctx, Preprocessor &PP,
                                            const syntax::TokenBuffer *Tokens);
ReferencedLocations findReferencedLocations(ParsedAST &AST);

struct ReferencedFiles {
  llvm::DenseSet<FileID> User;
  llvm::DenseSet<tooling::stdlib::Header> Stdlib;
  /// Files responsible for the symbols referenced in the main file and defined
  /// in private headers (private headers have IWYU pragma: private, include
  /// "public.h"). We store spelling of the public header (with quotes or angle
  /// brackets) files here to avoid dealing with full filenames and visibility.
  llvm::StringSet<> SpelledUmbrellas;
};

/// Retrieves IDs of all files containing SourceLocations from \p Locs.
/// The output only includes things SourceManager sees as files (not macro IDs).
/// This can include <built-in>, <scratch space> etc that are not true files.
/// \p HeaderResponsible returns the public header that should be included given
/// symbols from a file with the given FileID (example: public headers should be
/// preferred to non self-contained and private headers).
/// \p UmbrellaHeader returns the public public header is responsible for
/// providing symbols from a file with the given FileID (example: MyType.h
/// should be included instead of MyType_impl.h).
ReferencedFiles findReferencedFiles(
    const ReferencedLocations &Locs, const SourceManager &SM,
    llvm::function_ref<FileID(FileID)> HeaderResponsible,
    llvm::function_ref<std::optional<StringRef>(FileID)> UmbrellaHeader);
ReferencedFiles findReferencedFiles(const ReferencedLocations &Locs,
                                    const IncludeStructure &Includes,
                                    const CanonicalIncludes &CanonIncludes,
                                    const SourceManager &SM);

/// Maps FileIDs to the internal IncludeStructure representation (HeaderIDs).
/// FileIDs that are not true files (<built-in> etc) are dropped.
llvm::DenseSet<IncludeStructure::HeaderID>
translateToHeaderIDs(const ReferencedFiles &Files,
                     const IncludeStructure &Includes, const SourceManager &SM);

/// Retrieves headers that are referenced from the main file but not used.
/// In unclear cases, headers are not marked as unused.
std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles,
          const llvm::StringSet<> &ReferencedPublicHeaders);

std::vector<const Inclusion *> computeUnusedIncludes(ParsedAST &AST);
// The same as computeUnusedIncludes, but it is an experimental and
// include-cleaner-lib-based implementation.
std::vector<const Inclusion *>
computeUnusedIncludesExperimental(ParsedAST &AST);

std::vector<Diag> issueUnusedIncludesDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code);

/// Affects whether standard library includes should be considered for
/// removal. This is off by default for now due to implementation limitations:
/// - macros are not tracked
/// - symbol names without a unique associated header are not tracked
/// - references to std-namespaced C types are not properly tracked:
///   instead of std::size_t -> <cstddef> we see ::size_t -> <stddef.h>
/// FIXME: remove this hack once the implementation is good enough.
void setIncludeCleanerAnalyzesStdlib(bool B);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDECLEANER_H
