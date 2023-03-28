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
#include "clang-include-cleaner/Types.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include <tuple>
#include <vector>

namespace clang {
namespace clangd {

// Data needed for missing include diagnostics.
struct MissingIncludeDiagInfo {
  include_cleaner::Symbol Symbol;
  syntax::FileRange SymRefRange;
  std::vector<include_cleaner::Header> Providers;

  bool operator==(const MissingIncludeDiagInfo &Other) const {
    return std::tie(SymRefRange, Providers, Symbol) ==
           std::tie(Other.SymRefRange, Other.Providers, Other.Symbol);
  }
};

struct IncludeCleanerFindings {
  std::vector<const Inclusion *> UnusedIncludes;
  std::vector<MissingIncludeDiagInfo> MissingIncludes;
};

/// Retrieves headers that are referenced from the main file but not used.
/// In unclear cases, headers are not marked as unused.
std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles,
          const llvm::StringSet<> &ReferencedPublicHeaders);

IncludeCleanerFindings computeIncludeCleanerFindings(ParsedAST &AST);

std::vector<Diag> issueIncludeCleanerDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code);

/// Affects whether standard library includes should be considered for
/// removal. This is off by default for now due to implementation limitations:
/// - macros are not tracked
/// - symbol names without a unique associated header are not tracked
/// - references to std-namespaced C types are not properly tracked:
///   instead of std::size_t -> <cstddef> we see ::size_t -> <stddef.h>
/// FIXME: remove this hack once the implementation is good enough.
void setIncludeCleanerAnalyzesStdlib(bool B);

/// Converts the clangd include representation to include-cleaner
/// include representation.
include_cleaner::Includes
convertIncludes(const SourceManager &SM,
                const llvm::ArrayRef<Inclusion> Includes);

/// Determines the header spelling of an include-cleaner header
/// representation. The spelling contains the ""<> characters.
std::string spellHeader(ParsedAST &AST, const FileEntry *MainFile,
                        include_cleaner::Header Provider);

std::vector<include_cleaner::SymbolReference>
collectMacroReferences(ParsedAST &AST);
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDECLEANER_H
