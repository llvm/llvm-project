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

static bool MinCondition(const BinaryOperator::Opcode &Op, const Expr *CondLhs,
                         const Expr *CondRhs, const Expr *AssignLhs,
                         const Expr *AssignRhs, const ASTContext &Context) {
  if ((Op == BO_LT || Op == BO_LE) &&
      (tidy::utils::areStatementsIdentical(CondLhs, AssignRhs, Context) &&
       tidy::utils::areStatementsIdentical(CondRhs, AssignLhs, Context)))
    return true;

  if ((Op == BO_GT || Op == BO_GE) &&
      (tidy::utils::areStatementsIdentical(CondLhs, AssignLhs, Context) &&
       tidy::utils::areStatementsIdentical(CondRhs, AssignRhs, Context)))
    return true;

  return false;
}

static bool MaxCondition(const BinaryOperator::Opcode &Op, const Expr *CondLhs,
                         const Expr *CondRhs, const Expr *AssignLhs,
                         const Expr *AssignRhs, const ASTContext &Context) {
  if ((Op == BO_LT || Op == BO_LE) &&
      (tidy::utils::areStatementsIdentical(CondLhs, AssignLhs, Context) &&
       tidy::utils::areStatementsIdentical(CondRhs, AssignRhs, Context)))
    return true;

  if ((Op == BO_GT || Op == BO_GE) &&
      (tidy::utils::areStatementsIdentical(CondLhs, AssignRhs, Context) &&
       tidy::utils::areStatementsIdentical(CondRhs, AssignLhs, Context)))
    return true;

  return false;
}

static std::string
CreateReplacement(const bool UseMax, const BinaryOperator::Opcode &Op,
                  const Expr *CondLhs, const Expr *CondRhs,
                  const Expr *AssignLhs, const ASTContext &Context,
                  const SourceManager &Source, const LangOptions &LO,
                  const StringRef &FunctionName) {
  const auto CondLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondLhs->getSourceRange()), Source, LO);
  const auto CondRhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondRhs->getSourceRange()), Source, LO);
  const auto AssignLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(AssignLhs->getSourceRange()), Source, LO);
  return (AssignLhsStr + " = " + FunctionName +
          ((CondLhs->getType() != CondRhs->getType())
               ? "<" + AssignLhs->getType().getAsString() + ">("
               : "(") +
          CondLhsStr + ", " + CondRhsStr + ");")
      .str();
}

UseStdMinMaxCheck::UseStdMinMaxCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseStdMinMaxCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void UseStdMinMaxCheck::registerMatchers(MatchFinder *Finder) {
  auto AssignOperator =
      binaryOperator(hasOperatorName("="), hasLHS(expr().bind("AssignLhs")),
                     hasRHS(expr().bind("AssignRhs")));

  Finder->addMatcher(
      ifStmt(
          stmt().bind("if"),
          unless(hasElse(stmt())), // Ensure `if` has no `else`
          hasCondition(binaryOperator(hasAnyOperatorName("<", ">", "<=", ">="),
                                      hasLHS(expr().bind("CondLhs")),
                                      hasRHS(expr().bind("CondRhs")))
                           .bind("binaryOp")),
          hasThen(
              anyOf(stmt(AssignOperator),
                    compoundStmt(statementCountIs(1), has(AssignOperator)))),
          hasParent(stmt(unless(ifStmt(hasElse(
              equalsBoundNode("if"))))))), // Ensure `if` has no `else if`
      this);
}

void UseStdMinMaxCheck::registerPPCallbacks(const SourceManager &SM,
                                            Preprocessor *PP,
                                            Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseStdMinMaxCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto &Context = *Result.Context;
  const auto &LO = Context.getLangOpts();
  const auto *CondLhs = Result.Nodes.getNodeAs<Expr>("CondLhs");
  const auto *CondRhs = Result.Nodes.getNodeAs<Expr>("CondRhs");
  const auto *AssignLhs = Result.Nodes.getNodeAs<Expr>("AssignLhs");
  const auto *AssignRhs = Result.Nodes.getNodeAs<Expr>("AssignRhs");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("binaryOp");
  const auto BinaryOpcode = BinaryOp->getOpcode();
  const auto OperatorStr = BinaryOp->getOpcodeStr();
  const SourceManager &Source = Context.getSourceManager();
  const SourceLocation IfLocation = If->getIfLoc();
  const SourceLocation ThenLocation = If->getEndLoc();

  // Ignore Macros
  if (IfLocation.isMacroID() || ThenLocation.isMacroID())
    return;

  auto ReplaceAndDiagnose = [&](bool UseMax) {
    const llvm::StringRef FunctionName = UseMax ? "std::max" : "std::min";
    diag(IfLocation, "use `std::%0` instead of `%1`")
        << (UseMax ? "max" : "min") << OperatorStr
        << FixItHint::CreateReplacement(
               SourceRange(IfLocation, Lexer::getLocForEndOfToken(
                                           ThenLocation, 0, Source, LO)),
               CreateReplacement(UseMax, BinaryOpcode, CondLhs, CondRhs,
                                 AssignLhs, Context, Source, LO, FunctionName))
        << IncludeInserter.createIncludeInsertion(
               Source.getFileID(If->getBeginLoc()), AlgorithmHeader);
  };

  if (MinCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                   Context)) {
    ReplaceAndDiagnose(/*UseMax=*/false);
  } else if (MaxCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                          Context)) {
    ReplaceAndDiagnose(/*UseMax=*/true);
  }
}

} // namespace clang::tidy::readability
