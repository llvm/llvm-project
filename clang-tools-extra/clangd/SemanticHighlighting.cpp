//===--- SemanticHighlighting.cpp - ------------------------- ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SemanticHighlighting.h"
#include "Logger.h"
#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace clangd {
namespace {

// Collects all semantic tokens in an ASTContext.
class HighlightingTokenCollector
    : public RecursiveASTVisitor<HighlightingTokenCollector> {
  std::vector<HighlightingToken> Tokens;
  ASTContext &Ctx;
  const SourceManager &SM;

public:
  HighlightingTokenCollector(ParsedAST &AST)
      : Ctx(AST.getASTContext()), SM(AST.getSourceManager()) {}

  std::vector<HighlightingToken> collectTokens() {
    Tokens.clear();
    TraverseAST(Ctx);
    return Tokens;
  }

  bool VisitVarDecl(VarDecl *Var) {
    addToken(Var, HighlightingKind::Variable);
    return true;
  }
  bool VisitFunctionDecl(FunctionDecl *Func) {
    addToken(Func, HighlightingKind::Function);
    return true;
  }

private:
  void addToken(const NamedDecl *D, HighlightingKind Kind) {
    if (D->getLocation().isMacroID())
      // FIXME: skip tokens inside macros for now.
      return;

    if (D->getDeclName().isEmpty())
      // Don't add symbols that don't have any length.
      return;

    auto R = getTokenRange(SM, Ctx.getLangOpts(), D->getLocation());
    if (!R) {
      // R should always have a value, if it doesn't something is very wrong.
      elog("Tried to add semantic token with an invalid range");
      return;
    }

    Tokens.push_back({Kind, R.getValue()});
  }
};

} // namespace

bool operator==(const HighlightingToken &Lhs, const HighlightingToken &Rhs) {
  return Lhs.Kind == Rhs.Kind && Lhs.R == Rhs.R;
}

std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST) {
  return HighlightingTokenCollector(AST).collectTokens();
}

} // namespace clangd
} // namespace clang
