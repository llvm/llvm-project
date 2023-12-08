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
#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Token.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/TokenBufferTokenManager.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <queue>
#include <vector>

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

std::optional<FoldingRange> toFoldingRange(SourceRange SR,
                                           const SourceManager &SM) {
  const auto Begin = SM.getDecomposedLoc(SR.getBegin()),
             End = SM.getDecomposedLoc(SR.getEnd());
  // Do not produce folding ranges if either range ends is not within the main
  // file. Macros have their own FileID so this also checks if locations are not
  // within the macros.
  if ((Begin.first != SM.getMainFileID()) || (End.first != SM.getMainFileID()))
    return std::nullopt;
  FoldingRange Range;
  Range.startCharacter = SM.getColumnNumber(Begin.first, Begin.second) - 1;
  Range.startLine = SM.getLineNumber(Begin.first, Begin.second) - 1;
  Range.endCharacter = SM.getColumnNumber(End.first, End.second) - 1;
  Range.endLine = SM.getLineNumber(End.first, End.second) - 1;
  return Range;
}

std::optional<FoldingRange>
extractFoldingRange(const syntax::Node *Node,
                    const syntax::TokenBufferTokenManager &TM) {
  if (const auto *Stmt = dyn_cast<syntax::CompoundStatement>(Node)) {
    const auto *LBrace = cast_or_null<syntax::Leaf>(
        Stmt->findChild(syntax::NodeRole::OpenParen));
    // FIXME(kirillbobyrev): This should find the last child. Compound
    // statements have only one pair of braces so this is valid but for other
    // node kinds it might not be correct.
    const auto *RBrace = cast_or_null<syntax::Leaf>(
        Stmt->findChild(syntax::NodeRole::CloseParen));
    if (!LBrace || !RBrace)
      return std::nullopt;
    // Fold the entire range within braces, including whitespace.
    const SourceLocation LBraceLocInfo =
                             TM.getToken(LBrace->getTokenKey())->endLocation(),
                         RBraceLocInfo =
                             TM.getToken(RBrace->getTokenKey())->location();
    auto Range = toFoldingRange(SourceRange(LBraceLocInfo, RBraceLocInfo),
                                TM.sourceManager());
    // Do not generate folding range for compound statements without any
    // nodes and newlines.
    if (Range && Range->startLine != Range->endLine)
      return Range;
  }
  return std::nullopt;
}

// Traverse the tree and collect folding ranges along the way.
std::vector<FoldingRange>
collectFoldingRanges(const syntax::Node *Root,
                     const syntax::TokenBufferTokenManager &TM) {
  std::queue<const syntax::Node *> Nodes;
  Nodes.push(Root);
  std::vector<FoldingRange> Result;
  while (!Nodes.empty()) {
    const syntax::Node *Node = Nodes.front();
    Nodes.pop();
    const auto Range = extractFoldingRange(Node, TM);
    if (Range)
      Result.push_back(*Range);
    if (const auto *T = dyn_cast<syntax::Tree>(Node))
      for (const auto *NextNode = T->getFirstChild(); NextNode;
           NextNode = NextNode->getNextSibling())
        Nodes.push(NextNode);
  }
  return Result;
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
    if (!SR || SM.getFileID(SR->getBegin()) != SM.getMainFileID()) {
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
       llvm::MutableArrayRef(Ranges.data(), Ranges.size()).drop_front()) {
    Tail->parent = std::make_unique<SelectionRange>();
    Tail = Tail->parent.get();
    Tail->range = std::move(Range);
  }

  return std::move(Head);
}

// FIXME(kirillbobyrev): Collect comments, PP conditional regions, includes and
// other code regions (e.g. public/private/protected sections of classes,
// control flow statement bodies).
// Related issue: https://github.com/clangd/clangd/issues/310
llvm::Expected<std::vector<FoldingRange>> getFoldingRanges(ParsedAST &AST) {
  syntax::Arena A;
  syntax::TokenBufferTokenManager TM(AST.getTokens(), AST.getLangOpts(),
                                     AST.getSourceManager());
  const auto *SyntaxTree = syntax::buildSyntaxTree(A, TM, AST.getASTContext());
  return collectFoldingRanges(SyntaxTree, TM);
}

// FIXME( usaxena95): Collect PP conditional regions, includes and other code
// regions (e.g. public/private/protected sections of classes, control flow
// statement bodies).
// Related issue: https://github.com/clangd/clangd/issues/310
llvm::Expected<std::vector<FoldingRange>>
getFoldingRanges(const std::string &Code, bool LineFoldingOnly) {
  auto OrigStream = pseudo::lex(Code, clang::pseudo::genericLangOpts());

  auto DirectiveStructure = pseudo::DirectiveTree::parse(OrigStream);
  pseudo::chooseConditionalBranches(DirectiveStructure, OrigStream);

  // FIXME: Provide ranges in the disabled-PP regions as well.
  auto Preprocessed = DirectiveStructure.stripDirectives(OrigStream);

  auto ParseableStream = cook(Preprocessed, clang::pseudo::genericLangOpts());
  pseudo::pairBrackets(ParseableStream);

  std::vector<FoldingRange> Result;
  auto AddFoldingRange = [&](Position Start, Position End,
                             llvm::StringLiteral Kind) {
    if (Start.line >= End.line)
      return;
    FoldingRange FR;
    FR.startLine = Start.line;
    FR.startCharacter = Start.character;
    FR.endLine = End.line;
    FR.endCharacter = End.character;
    FR.kind = Kind.str();
    Result.push_back(FR);
  };
  auto OriginalToken = [&](const pseudo::Token &T) {
    return OrigStream.tokens()[T.OriginalIndex];
  };
  auto StartOffset = [&](const pseudo::Token &T) {
    return OriginalToken(T).text().data() - Code.data();
  };
  auto StartPosition = [&](const pseudo::Token &T) {
    return offsetToPosition(Code, StartOffset(T));
  };
  auto EndOffset = [&](const pseudo::Token &T) {
    return StartOffset(T) + OriginalToken(T).Length;
  };
  auto EndPosition = [&](const pseudo::Token &T) {
    return offsetToPosition(Code, EndOffset(T));
  };
  auto Tokens = ParseableStream.tokens();
  // Brackets.
  for (const auto &Tok : Tokens) {
    if (auto *Paired = Tok.pair()) {
      // Process only token at the start of the range. Avoid ranges on a single
      // line.
      if (Tok.Line < Paired->Line) {
        Position Start = offsetToPosition(Code, 1 + StartOffset(Tok));
        Position End = StartPosition(*Paired);
        if (LineFoldingOnly)
          End.line--;
        AddFoldingRange(Start, End, FoldingRange::REGION_KIND);
      }
    }
  }
  auto IsBlockComment = [&](const pseudo::Token &T) {
    assert(T.Kind == tok::comment);
    return OriginalToken(T).Length >= 2 &&
           Code.substr(StartOffset(T), 2) == "/*";
  };
  // Multi-line comments.
  for (auto *T = Tokens.begin(); T != Tokens.end();) {
    if (T->Kind != tok::comment) {
      T++;
      continue;
    }
    pseudo::Token *FirstComment = T;
    // Show starting sentinals (// and /*) of the comment.
    Position Start = offsetToPosition(Code, 2 + StartOffset(*FirstComment));
    pseudo::Token *LastComment = T;
    Position End = EndPosition(*T);
    while (T != Tokens.end() && T->Kind == tok::comment &&
           StartPosition(*T).line <= End.line + 1) {
      End = EndPosition(*T);
      LastComment = T;
      T++;
    }
    if (IsBlockComment(*FirstComment)) {
      if (LineFoldingOnly)
        // Show last line of a block comment.
        End.line--;
      if (IsBlockComment(*LastComment))
        // Show ending sentinal "*/" of the block comment.
        End.character -= 2;
    }
    AddFoldingRange(Start, End, FoldingRange::COMMENT_KIND);
  }
  return Result;
}

} // namespace clangd
} // namespace clang
