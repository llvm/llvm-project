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

#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "clang-include-cleaner/Types.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>
#include <string>
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

IncludeCleanerFindings
computeIncludeCleanerFindings(ParsedAST &AST,
                              bool AnalyzeAngledIncludes = false);

using HeaderFilter = llvm::ArrayRef<std::function<bool(llvm::StringRef)>>;
std::vector<Diag>
issueIncludeCleanerDiagnostics(ParsedAST &AST, llvm::StringRef Code,
                               const IncludeCleanerFindings &Findings,
                               const ThreadsafeFS &TFS,
                               HeaderFilter IgnoreHeader = {});

/// Converts the clangd include representation to include-cleaner
/// include representation.
include_cleaner::Includes convertIncludes(const ParsedAST &);

std::vector<include_cleaner::SymbolReference>
collectMacroReferences(ParsedAST &AST);

/// Whether this #include is considered to provide a particular symbol.
///
/// This means it satisfies the reference, and no other #include does better.
/// `Providers` is the symbol's candidate headers according to walkUsed().
bool isPreferredProvider(const Inclusion &, const include_cleaner::Includes &,
                         llvm::ArrayRef<include_cleaner::Header> Providers);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDECLEANER_H
