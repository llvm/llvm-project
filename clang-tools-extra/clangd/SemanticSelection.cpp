//===--- SemanticSelection.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SemanticSelection.h"
#include "FindSymbols.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
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

// Recursively collects FoldingRange from a symbol and its children.
void collectFoldingRanges(DocumentSymbol Symbol,
                          std::vector<FoldingRange> &Result) {
  FoldingRange Range;
  Range.startLine = Symbol.range.start.line;
  Range.startCharacter = Symbol.range.start.character;
  Range.endLine = Symbol.range.end.line;
  Range.endCharacter = Symbol.range.end.character;
  Result.push_back(Range);
  for (const auto &Child : Symbol.children)
    collectFoldingRanges(Child, Result);
}

} // namespace

llvm::Expected<SelectionRange> getSemanticRanges(ParsedAST &AST, Position Pos) {
  std::vector<Range> Ranges;
  const auto &SM = AST.getSourceManager();
  const auto &LangOpts = AST.getLangOpts();

  auto FID = SM.getMainFileID();
  auto Offset = positionToOffset(SM.getBufferData(FID), Pos);
  if (!Offset) {
    return Offset.takeError();
  }

  // Get node under the cursor.
  SelectionTree ST = SelectionTree::createRight(
      AST.getASTContext(), AST.getTokens(), *Offset, *Offset);
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
    addIfDistinct(R, Ranges);
  }

  if (Ranges.empty()) {
    // LSP provides no way to signal "the point is not within a semantic range".
    // Return an empty range at the point.
    SelectionRange Empty;
    Empty.range.start = Empty.range.end = Pos;
    return std::move(Empty);
  }

  // Convert to the LSP linked-list representation.
  SelectionRange Head;
  Head.range = std::move(Ranges.front());
  SelectionRange *Tail = &Head;
  for (auto &Range :
       llvm::makeMutableArrayRef(Ranges.data(), Ranges.size()).drop_front()) {
    Tail->parent = std::make_unique<SelectionRange>();
    Tail = Tail->parent.get();
    Tail->range = std::move(Range);
  }

  return std::move(Head);
}

// FIXME(kirillbobyrev): Collect comments, PP conditional regions, includes and
// other code regions (e.g. public/private/protected sections of classes,
// control flow statement bodies).
// Related issue:
// https://github.com/clangd/clangd/issues/310
llvm::Expected<std::vector<FoldingRange>> getFoldingRanges(ParsedAST &AST) {
  // FIXME(kirillbobyrev): getDocumentSymbols() is conveniently available but
  // limited (e.g. doesn't yield blocks inside functions and provides ranges for
  // nodes themselves instead of their contents which is less useful). Replace
  // this with a more general RecursiveASTVisitor implementation instead.
  auto DocumentSymbols = getDocumentSymbols(AST);
  if (!DocumentSymbols)
    return DocumentSymbols.takeError();
  std::vector<FoldingRange> Result;
  for (const auto &Symbol : *DocumentSymbols)
    collectFoldingRanges(Symbol, Result);
  return Result;
}

} // namespace clangd
} // namespace clang
