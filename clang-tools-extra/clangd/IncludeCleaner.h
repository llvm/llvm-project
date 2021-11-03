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
/// FIXME(kirillbobyrev): Add support for IWYU pragmas.
/// FIXME(kirillbobyrev): Add support for standard library headers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_CLEANER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_CLEANER_H

#include "Headers.h"
#include "ParsedAST.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>

namespace clang {
namespace clangd {

using ReferencedLocations = llvm::DenseSet<SourceLocation>;
/// Finds locations of all symbols used in the main file.
///
/// - RecursiveASTVisitor finds references to symbols and records their
///   associated locations. These may be macro expansions, and are not resolved
///   to their spelling or expansion location. These locations are later used to
///   determine which headers should be marked as "used" and "directly used".
/// - We also examine all identifier tokens in the file in case they reference
///   macros.
///
/// We use this to compute unused headers, so we:
///
/// - cover the whole file in a single traversal for efficiency
/// - don't attempt to describe where symbols were referenced from in
///   ambiguous cases (e.g. implicitly used symbols, multiple declarations)
/// - err on the side of reporting all possible locations
ReferencedLocations findReferencedLocations(ParsedAST &AST);

/// Retrieves IDs of all files containing SourceLocations from \p Locs.
/// The output only includes things SourceManager sees as files (not macro IDs).
/// This can include <built-in>, <scratch space> etc that are not true files.
llvm::DenseSet<FileID> findReferencedFiles(const ReferencedLocations &Locs,
                                           const SourceManager &SM);

/// Maps FileIDs to the internal IncludeStructure representation (HeaderIDs).
/// FileIDs that are not true files (<built-in> etc) are dropped.
llvm::DenseSet<IncludeStructure::HeaderID>
translateToHeaderIDs(const llvm::DenseSet<FileID> &Files,
                     const IncludeStructure &Includes, const SourceManager &SM);

/// Retrieves headers that are referenced from the main file but not used.
/// In unclear cases, headers are not marked as unused.
std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles);

std::vector<const Inclusion *> computeUnusedIncludes(ParsedAST &AST);

std::vector<Diag> issueUnusedIncludesDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_CLEANER_H
