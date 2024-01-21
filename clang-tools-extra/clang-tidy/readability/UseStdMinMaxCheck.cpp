//===--- UseStdMinMaxCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdMinMaxCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static const llvm::StringRef AlgorithmHeader("<algorithm>");

UseStdMinMaxCheck::UseStdMinMaxCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseStdMinMaxCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void UseStdMinMaxCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(unless(hasAncestor(ifStmt())),
          hasCondition(binaryOperator(hasAnyOperatorName("<", ">", "<=", ">="),
                                      hasLHS(expr().bind("CondLhs")),
                                      hasRHS(expr().bind("CondRhs")))),
          hasThen(anyOf(stmt(binaryOperator(hasOperatorName("="),
                                            hasLHS(expr().bind("AssignLhs")),
                                            hasRHS(expr().bind("AssignRhs")))),
                        compoundStmt(statementCountIs(1),
                                     has(binaryOperator(
                                         hasOperatorName("="),
                                         hasLHS(expr().bind("AssignLhs")),
                                         hasRHS(expr().bind("AssignRhs"))))))))
          .bind("if"),
      this);
}

void UseStdMinMaxCheck::registerPPCallbacks(const SourceManager &SM,
                                            Preprocessor *PP,
                                            Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseStdMinMaxCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CondLhs = Result.Nodes.getNodeAs<Expr>("CondLhs");
  const auto *CondRhs = Result.Nodes.getNodeAs<Expr>("CondRhs");
  const auto *AssignLhs = Result.Nodes.getNodeAs<Expr>("AssignLhs");
  const auto *AssignRhs = Result.Nodes.getNodeAs<Expr>("AssignRhs");
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto &Context = *Result.Context;
  const auto &LO = Context.getLangOpts();
  const SourceManager &Source = Context.getSourceManager();

  const auto *BinaryOp = dyn_cast<BinaryOperator>(If->getCond());
  if (!BinaryOp || If->hasElseStorage())
    return;

  const SourceLocation IfLocation = If->getIfLoc();
  const SourceLocation ThenLocation = If->getEndLoc();

  if (IfLocation.isMacroID() || ThenLocation.isMacroID())
    return;

  const auto CreateReplacement = [&](bool UseMax) {
    const auto *FunctionName = UseMax ? "std::max" : "std::min";
    const auto CondLhsStr = Lexer::getSourceText(
        Source.getExpansionRange(CondLhs->getSourceRange()), Source, LO);
    const auto CondRhsStr = Lexer::getSourceText(
        Source.getExpansionRange(CondRhs->getSourceRange()), Source, LO);
    const auto AssignLhsStr = Lexer::getSourceText(
        Source.getExpansionRange(AssignLhs->getSourceRange()), Source, LO);
    if (CondLhs->getType() != CondRhs->getType()) {
      return (AssignLhsStr + " = " + FunctionName + "<" +
              AssignLhs->getType().getAsString() + ">(" + CondLhsStr + ", " +
              CondRhsStr + ");")
          .str();
    } else {
      return (AssignLhsStr + " = " + FunctionName + "(" + CondLhsStr + ", " +
              CondRhsStr + ");")
          .str();
    }
  };
  const auto OperatorStr = BinaryOp->getOpcodeStr();
  if (((BinaryOp->getOpcode() == BO_LT || BinaryOp->getOpcode() == BO_LE) &&
       (tidy::utils::areStatementsIdentical(CondLhs, AssignRhs, Context) &&
        tidy::utils::areStatementsIdentical(CondRhs, AssignLhs, Context))) ||
      ((BinaryOp->getOpcode() == BO_GT || BinaryOp->getOpcode() == BO_GE) &&
       (tidy::utils::areStatementsIdentical(CondLhs, AssignLhs, Context) &&
        tidy::utils::areStatementsIdentical(CondRhs, AssignRhs, Context)))) {
    diag(IfLocation, "use `std::min` instead of `%0`")
        << OperatorStr
        << FixItHint::CreateReplacement(
               SourceRange(IfLocation, Lexer::getLocForEndOfToken(
                                           ThenLocation, 0, Source, LO)),
               CreateReplacement(/*UseMax=*/false))
        << IncludeInserter.createIncludeInsertion(
               Source.getFileID(If->getBeginLoc()), AlgorithmHeader);

  } else if (((BinaryOp->getOpcode() == BO_LT ||
               BinaryOp->getOpcode() == BO_LE) &&
              (tidy::utils::areStatementsIdentical(CondLhs, AssignLhs,
                                                   Context) &&
               tidy::utils::areStatementsIdentical(CondRhs, AssignRhs,
                                                   Context))) ||
             ((BinaryOp->getOpcode() == BO_GT ||
               BinaryOp->getOpcode() == BO_GE) &&
              (tidy::utils::areStatementsIdentical(CondLhs, AssignRhs,
                                                   Context) &&
               tidy::utils::areStatementsIdentical(CondRhs, AssignLhs,
                                                   Context)))) {
    diag(IfLocation, "use `std::max` instead of `%0`")
        << OperatorStr
        << FixItHint::CreateReplacement(
               SourceRange(IfLocation, Lexer::getLocForEndOfToken(
                                           ThenLocation, 0, Source, LO)),
               CreateReplacement(/*UseMax=*/true))
        << IncludeInserter.createIncludeInsertion(
               Source.getFileID(If->getBeginLoc()), AlgorithmHeader);
  }
}

} // namespace clang::tidy::readability
