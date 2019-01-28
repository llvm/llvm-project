//===--- IncludeFixer.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_FIXER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_FIXER_H

#include "Diagnostics.h"
#include "Headers.h"
#include "index/Index.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
namespace clangd {

/// Attempts to recover from error diagnostics by suggesting include insertion
/// fixes. For example, member access into incomplete type can be fixes by
/// include headers with the definition.
class IncludeFixer {
public:
  IncludeFixer(llvm::StringRef File, std::shared_ptr<IncludeInserter> Inserter,
               const SymbolIndex &Index, unsigned IndexRequestLimit)
      : File(File), Inserter(std::move(Inserter)), Index(Index),
        IndexRequestLimit(IndexRequestLimit) {}

  /// Returns include insertions that can potentially recover the diagnostic.
  std::vector<Fix> fix(DiagnosticsEngine::Level DiagLevel,
                       const clang::Diagnostic &Info) const;

private:
  /// Attempts to recover diagnostic caused by an incomplete type \p T.
  std::vector<Fix> fixIncompleteType(const Type &T) const;

  /// Generates header insertion fixes for \p Sym.
  std::vector<Fix> fixesForSymbol(const Symbol &Sym) const;

  std::string File;
  std::shared_ptr<IncludeInserter> Inserter;
  const SymbolIndex &Index;
  const unsigned IndexRequestLimit; // Make at most 5 index requests.
  mutable unsigned IndexRequestCount = 0;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDE_FIXER_H
