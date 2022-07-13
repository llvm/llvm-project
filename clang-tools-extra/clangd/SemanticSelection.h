//===--- SemanticSelection.h -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Features for giving interesting semantic ranges around the cursor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICSELECTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICSELECTION_H
#include "ParsedAST.h"
#include "Protocol.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>
namespace clang {
namespace clangd {

/// Returns the list of all interesting ranges around the Position \p Pos.
/// The interesting ranges corresponds to the AST nodes in the SelectionTree
/// containing \p Pos.
/// If pos is not in any interesting range, return [Pos, Pos).
llvm::Expected<SelectionRange> getSemanticRanges(ParsedAST &AST, Position Pos);

/// Returns a list of ranges whose contents might be collapsible in an editor.
/// This should include large scopes, preprocessor blocks etc.
llvm::Expected<std::vector<FoldingRange>> getFoldingRanges(ParsedAST &AST);

/// Returns a list of ranges whose contents might be collapsible in an editor.
/// This version uses the pseudoparser which does not require the AST.
llvm::Expected<std::vector<FoldingRange>>
getFoldingRanges(const std::string &Code);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICSELECTION_H
