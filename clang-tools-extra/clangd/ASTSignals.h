//===--- ASTSignals.h --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_ASTSIGNALS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_ASTSIGNALS_H

#include "ParsedAST.h"
#include "index/SymbolID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
namespace clangd {

/// Signals derived from a valid AST of a file.
/// Provides information that can only be extracted from the AST to actions that
/// can't access an AST. The signals are computed and updated asynchronously by
/// the ASTWorker and thus they are always stale and also can be absent.
/// Example usage: Information about the declarations used in a file affects
/// code-completion ranking in that file.
struct ASTSignals {
  /// Number of occurrences of each symbol present in the file.
  llvm::DenseMap<SymbolID, unsigned> ReferencedSymbols;
  /// Namespaces whose symbols are used in the file, and the number of such
  /// distinct symbols.
  llvm::StringMap<unsigned> RelatedNamespaces;

  static ASTSignals derive(const ParsedAST &AST);
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_ASTSIGNALS_H
