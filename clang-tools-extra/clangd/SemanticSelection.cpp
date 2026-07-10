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
#include "support/Bracket.h"
#include "support/DirectiveTree.h"
#include "support/Token.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
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

class PragmaRegionFinder {
  // Record the token range of a region:
  //
  //   #pragma region name[[
  //   ...
  //   ]]#pragma endregion
  std::vector<Token::Range> &Ranges;
  const TokenStream &Code;
  // Stack of starting token (the name of the region) indices for nested #pragma
  // region.
  std::vector<unsigned> Stack;

public:
  PragmaRegionFinder(std::vector<Token::Range> &Ranges, const TokenStream &Code)
      : Ranges(Ranges), Code(Code) {}

  void walk(const DirectiveTree &T) {
    for (const auto &C : T.Chunks)
      std::visit(*this, C);
  }

  void operator()(const DirectiveTree::Code &C) {}

  void operator()(const DirectiveTree::Directive &D) {
    // Get the tokens that make up this directive.
    auto Tokens = Code.tokens(D.Tokens);
    if (Tokens.empty())
      return;
    const Token &HashToken = Tokens.front();
    assert(HashToken.Kind == tok::hash);
    const Token &Pragma = HashToken.nextNC();
    if (Pragma.text() != "pragma")
      return;
    const Token &Value = Pragma.nextNC();

    // Handle "#pragma region name"
    if (Value.text() == "region") {
      // Find the last token at the same line.
      const Token *T = &Value.next();
      while (T < Tokens.end() && T->Line == Pragma.Line)
        T = &T->next();
      --T;
      Stack.push_back(T->OriginalIndex);
      return;
    }

    // Handle "#pragma endregion"
    if (Value.text() == "endregion") {
      if (Stack.empty())
        return; // unmatched end region; ignore.

      unsigned StartIdx = Stack.back();
      Stack.pop_back();
      Ranges.push_back(Token::Range{StartIdx, HashToken.OriginalIndex});
    }
  }

  void operator()(const DirectiveTree::Conditional &C) {
    // C.Branches needs to see the DirectiveTree definition, otherwise build
    // fails in C++20.
    [[maybe_unused]] DirectiveTree Dummy;
    for (const auto &[_, SubTree] : C.Branches)
      walk(SubTree);
  }
};

// FIXME( usaxena95): Collect includes and other code regions (e.g.
// public/private/protected sections of classes, control flow statement bodies).
// Related issue: https://github.com/clangd/clangd/issues/310
llvm::Expected<std::vector<FoldingRange>>
getFoldingRanges(const std::string &Code, bool LineFoldingOnly) {
  auto OrigStream = lex(Code, genericLangOpts());

  auto DirectiveStructure = DirectiveTree::parse(OrigStream);
  chooseConditionalBranches(DirectiveStructure, OrigStream);

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
  auto OriginalToken = [&](const Token &T) {
    return OrigStream.tokens()[T.OriginalIndex];
  };
  auto StartOffset = [&](const Token &T) {
    return OriginalToken(T).text().data() - Code.data();
  };
  auto StartPosition = [&](const Token &T) {
    return offsetToPosition(Code, StartOffset(T));
  };
  auto EndOffset = [&](const Token &T) {
    return StartOffset(T) + OriginalToken(T).Length;
  };
  auto EndPosition = [&](const Token &T) {
    return offsetToPosition(Code, EndOffset(T));
  };

  // Preprocessor directives
  auto PPRanges = pairDirectiveRanges(DirectiveStructure, OrigStream);
  for (const auto &R : PPRanges) {
    auto BTok = OrigStream.tokens()[R.Begin];
    auto ETok = OrigStream.tokens()[R.End];
    if (ETok.Kind == tok::eof)
      continue;
    if (BTok.Line >= ETok.Line)
      continue;

    Position Start = EndPosition(BTok);
    Position End = StartPosition(ETok);
    if (LineFoldingOnly)
      End.line--;
    AddFoldingRange(Start, End, FoldingRange::REGION_KIND);
  }

  // FIXME: Provide ranges in the disabled-PP regions as well.
  auto Preprocessed = DirectiveStructure.stripDirectives(OrigStream);

  auto ParseableStream = cook(Preprocessed, genericLangOpts());
  pairBrackets(ParseableStream);

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
  auto IsBlockComment = [&](const Token &T) {
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
    Token *FirstComment = T;
    // Show starting sentinals (// and /*) of the comment.
    Position Start = offsetToPosition(Code, 2 + StartOffset(*FirstComment));
    Token *LastComment = T;
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

  // #pragma region
  std::vector<Token::Range> Ranges;
  PragmaRegionFinder(Ranges, OrigStream).walk(DirectiveStructure);
  auto Ts = OrigStream.tokens();
  for (const auto &R : Ranges) {
    auto End = StartPosition(Ts[R.End]);
    if (LineFoldingOnly)
      End.line--;
    AddFoldingRange(EndPosition(Ts[R.Begin]), End, FoldingRange::REGION_KIND);
  }
  return Result;
}

} // namespace clangd
} // namespace clang
