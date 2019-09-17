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
#include <vector>
namespace clang {
namespace clangd {

/// Returns the list of all interesting ranges around the Position \p Pos.
/// The interesting ranges corresponds to the AST nodes in the SelectionTree
/// containing \p Pos.
/// Any range in the result strictly contains all the previous ranges in the
/// result. front() is the inner most range. back() is the outermost range.
llvm::Expected<std::vector<Range>> getSemanticRanges(ParsedAST &AST,
                                                     Position Pos);
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICSELECTION_H
