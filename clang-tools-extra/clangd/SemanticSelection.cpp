//===--- SemanticSelection.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SemanticSelection.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {
// Adds Range \p R to the Result if it is distinct from the last added Range.
// Assumes that only consecutive ranges can coincide.
void addIfDistinct(const Range &R, std::vector<Range> &Result) {
  if (Result.empty() || Result.back() != R) {
    Result.push_back(R);
  }
}
} // namespace

llvm::Expected<std::vector<Range>> getSemanticRanges(ParsedAST &AST,
                                                     Position Pos) {
  std::vector<Range> Result;
  const auto &SM = AST.getSourceManager();
  const auto &LangOpts = AST.getLangOpts();

  auto FID = SM.getMainFileID();
  auto Offset = positionToOffset(SM.getBufferData(FID), Pos);
  if (!Offset) {
    return Offset.takeError();
  }

  // Get node under the cursor.
  SelectionTree ST(AST.getASTContext(), AST.getTokens(), *Offset);
  for (const auto *Node = ST.commonAncestor(); Node != nullptr;
       Node = Node->Parent) {
    if (const Decl *D = Node->ASTNode.get<Decl>()) {
      if (llvm::isa<TranslationUnitDecl>(D)) {
        break;
      }
    }

    auto SR = toHalfOpenFileRange(SM, LangOpts, Node->ASTNode.getSourceRange());
    if (!SR.hasValue() || SM.getFileID(SR->getBegin()) != SM.getMainFileID()) {
      continue;
    }
    Range R;
    R.start = sourceLocToPosition(SM, SR->getBegin());
    R.end = sourceLocToPosition(SM, SR->getEnd());
    addIfDistinct(R, Result);
  }
  return Result;
}

} // namespace clangd
} // namespace clang
