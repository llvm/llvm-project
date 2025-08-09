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

namespace {

// Ignore if statements that are inside macros.
AST_MATCHER(IfStmt, isIfInMacro) {
  return Node.getIfLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

} // namespace

static const llvm::StringRef AlgorithmHeader("<algorithm>");

static bool minCondition(const BinaryOperator::Opcode Op, const Expr *CondLhs,
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

static bool maxCondition(const BinaryOperator::Opcode Op, const Expr *CondLhs,
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

static QualType getNonTemplateAlias(QualType QT) {
  while (true) {
    // cast to a TypedefType
    if (const TypedefType *TT = dyn_cast<TypedefType>(QT)) {
      // check if the typedef is a template and if it is dependent
      if (!TT->getDecl()->getDescribedTemplate() &&
          !TT->getDecl()->getDeclContext()->isDependentContext())
        return QT;
      QT = TT->desugar();
    } else {
      break;
    }
  }
  return QT;
}

static QualType getReplacementCastType(const Expr *CondLhs, const Expr *CondRhs,
                                       QualType ComparedType) {
  QualType LhsType = CondLhs->getType();
  QualType RhsType = CondRhs->getType();
  QualType LhsCanonicalType =
      LhsType.getCanonicalType().getNonReferenceType().getUnqualifiedType();
  QualType RhsCanonicalType =
      RhsType.getCanonicalType().getNonReferenceType().getUnqualifiedType();
  QualType GlobalImplicitCastType;
  if (LhsCanonicalType != RhsCanonicalType) {
    if (llvm::isa<IntegerLiteral>(CondRhs)) {
      GlobalImplicitCastType = getNonTemplateAlias(LhsType);
    } else if (llvm::isa<IntegerLiteral>(CondLhs)) {
      GlobalImplicitCastType = getNonTemplateAlias(RhsType);
    } else {
      GlobalImplicitCastType = getNonTemplateAlias(ComparedType);
    }
  }
  return GlobalImplicitCastType;
}

static std::string createReplacement(const Expr *CondLhs, const Expr *CondRhs,
                                     const Expr *AssignLhs,
                                     const SourceManager &Source,
                                     const LangOptions &LO,
                                     StringRef FunctionName,
                                     const BinaryOperator *BO) {
  const llvm::StringRef CondLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondLhs->getSourceRange()), Source, LO);
  const llvm::StringRef CondRhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondRhs->getSourceRange()), Source, LO);
  const llvm::StringRef AssignLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(AssignLhs->getSourceRange()), Source, LO);

  QualType GlobalImplicitCastType =
      getReplacementCastType(CondLhs, CondRhs, BO->getLHS()->getType());

  return (AssignLhsStr + " = " + FunctionName +
          (!GlobalImplicitCastType.isNull()
               ? "<" + GlobalImplicitCastType.getAsString() + ">("
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
      binaryOperator(hasOperatorName("="),
                     hasLHS(expr(unless(isTypeDependent())).bind("AssignLhs")),
                     hasRHS(expr(unless(isTypeDependent())).bind("AssignRhs")));
  auto BinaryOperator =
      binaryOperator(hasAnyOperatorName("<", ">", "<=", ">="),
                     hasLHS(expr(unless(isTypeDependent())).bind("CondLhs")),
                     hasRHS(expr(unless(isTypeDependent())).bind("CondRhs")))
          .bind("binaryOp");
  Finder->addMatcher(
      ifStmt(stmt().bind("if"), unless(isIfInMacro()),
             unless(hasElse(stmt())), // Ensure `if` has no `else`
             hasCondition(BinaryOperator),
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
  const clang::LangOptions &LO = Result.Context->getLangOpts();
  const auto *CondLhs = Result.Nodes.getNodeAs<Expr>("CondLhs");
  const auto *CondRhs = Result.Nodes.getNodeAs<Expr>("CondRhs");
  const auto *AssignLhs = Result.Nodes.getNodeAs<Expr>("AssignLhs");
  const auto *AssignRhs = Result.Nodes.getNodeAs<Expr>("AssignRhs");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("binaryOp");
  const clang::BinaryOperatorKind BinaryOpcode = BinaryOp->getOpcode();
  const SourceLocation IfLocation = If->getIfLoc();
  const SourceLocation ThenLocation = If->getEndLoc();

  auto ReplaceAndDiagnose = [&](const llvm::StringRef FunctionName) {
    const SourceManager &Source = *Result.SourceManager;
    diag(IfLocation, "use `%0` instead of `%1`")
        << FunctionName << BinaryOp->getOpcodeStr()
        << FixItHint::CreateReplacement(
               SourceRange(IfLocation, Lexer::getLocForEndOfToken(
                                           ThenLocation, 0, Source, LO)),
               createReplacement(CondLhs, CondRhs, AssignLhs, Source, LO,
                                 FunctionName, BinaryOp))
        << IncludeInserter.createIncludeInsertion(
               Source.getFileID(If->getBeginLoc()), AlgorithmHeader);
  };

  if (minCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                   (*Result.Context))) {
    ReplaceAndDiagnose("std::min");
  } else if (maxCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                          (*Result.Context))) {
    ReplaceAndDiagnose("std::max");
  }
}

} // namespace clang::tidy::readability
