//===--- SwapIfBranches.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ClangdUnit.h"
#include "Logger.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {
/// Swaps the 'then' and 'else' branch of the if statement.
/// Before:
///   if (foo) { return 10; } else { continue; }
///   ^^^^^^^                 ^^^^
/// After:
///   if (foo) { continue; } else { return 10; }
class SwapIfBranches : public Tweak {
public:
  const char *id() const override final;

  bool prepare(const Selection &Inputs) override;
  Expected<tooling::Replacements> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  IfStmt *If = nullptr;
};

REGISTER_TWEAK(SwapIfBranches);

class FindIfUnderCursor : public RecursiveASTVisitor<FindIfUnderCursor> {
public:
  FindIfUnderCursor(ASTContext &Ctx, SourceLocation CursorLoc, IfStmt *&Result)
      : Ctx(Ctx), CursorLoc(CursorLoc), Result(Result) {}

  bool VisitIfStmt(IfStmt *If) {
    // Check if the cursor is in the range of 'if (cond)'.
    // FIXME: this does not contain the closing paren, add it too.
    auto R = toHalfOpenFileRange(
        Ctx.getSourceManager(), Ctx.getLangOpts(),
        SourceRange(If->getIfLoc(), If->getCond()->getEndLoc().isValid()
                                        ? If->getCond()->getEndLoc()
                                        : If->getIfLoc()));
    if (R && halfOpenRangeTouches(Ctx.getSourceManager(), *R, CursorLoc)) {
      Result = If;
      return false;
    }
    // Check the range of 'else'.
    R = toHalfOpenFileRange(Ctx.getSourceManager(), Ctx.getLangOpts(),
                            SourceRange(If->getElseLoc()));
    if (R && halfOpenRangeTouches(Ctx.getSourceManager(), *R, CursorLoc)) {
      Result = If;
      return false;
    }

    return true;
  }

private:
  ASTContext &Ctx;
  SourceLocation CursorLoc;
  IfStmt *&Result;
};
} // namespace

bool SwapIfBranches::prepare(const Selection &Inputs) {
  auto &Ctx = Inputs.AST.getASTContext();
  FindIfUnderCursor(Ctx, Inputs.Cursor, If).TraverseAST(Ctx);
  if (!If)
    return false;

  // avoid dealing with single-statement brances, they require careful handling
  // to avoid changing semantics of the code (i.e. dangling else).
  if (!If->getThen() || !llvm::isa<CompoundStmt>(If->getThen()) ||
      !If->getElse() || !llvm::isa<CompoundStmt>(If->getElse()))
    return false;
  return true;
}

Expected<tooling::Replacements> SwapIfBranches::apply(const Selection &Inputs) {
  auto &Ctx = Inputs.AST.getASTContext();
  auto &SrcMgr = Ctx.getSourceManager();

  auto ThenRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                     If->getThen()->getSourceRange());
  if (!ThenRng)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Could not obtain range of the 'then' branch. Macros?");
  auto ElseRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                     If->getElse()->getSourceRange());
  if (!ElseRng)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Could not obtain range of the 'else' branch. Macros?");

  auto ThenCode = toSourceCode(SrcMgr, *ThenRng);
  auto ElseCode = toSourceCode(SrcMgr, *ElseRng);

  tooling::Replacements Result;
  if (auto Err = Result.add(tooling::Replacement(Ctx.getSourceManager(),
                                                 ThenRng->getBegin(),
                                                 ThenCode.size(), ElseCode)))
    return std::move(Err);
  if (auto Err = Result.add(tooling::Replacement(Ctx.getSourceManager(),
                                                 ElseRng->getBegin(),
                                                 ElseCode.size(), ThenCode)))
    return std::move(Err);
  return Result;
}

std::string SwapIfBranches::title() const { return "Swap if branches"; }
} // namespace clangd
} // namespace clang
