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
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static bool isImplicitCastType(const clang::CastKind castKind) {
  switch (castKind) {
  case clang::CK_CPointerToObjCPointerCast:
  case clang::CK_BlockPointerToObjCPointerCast:
  case clang::CK_BitCast:
  case clang::CK_AnyPointerToBlockPointerCast:
  case clang::CK_NullToMemberPointer:
  case clang::CK_NullToPointer:
  case clang::CK_IntegralToPointer:
  case clang::CK_PointerToIntegral:
  case clang::CK_IntegralCast:
  case clang::CK_BooleanToSignedIntegral:
  case clang::CK_IntegralToFloating:
  case clang::CK_FloatingToIntegral:
  case clang::CK_FloatingCast:
  case clang::CK_ObjCObjectLValueCast:
  case clang::CK_FloatingRealToComplex:
  case clang::CK_FloatingComplexToReal:
  case clang::CK_FloatingComplexCast:
  case clang::CK_FloatingComplexToIntegralComplex:
  case clang::CK_IntegralRealToComplex:
  case clang::CK_IntegralComplexToReal:
  case clang::CK_IntegralComplexCast:
  case clang::CK_IntegralComplexToFloatingComplex:
  case clang::CK_FloatingToFixedPoint:
  case clang::CK_FixedPointToFloating:
  case clang::CK_FixedPointCast:
  case clang::CK_FixedPointToIntegral:
  case clang::CK_IntegralToFixedPoint:
  case clang::CK_MatrixCast:
  case clang::CK_PointerToBoolean:
  case clang::CK_IntegralToBoolean:
  case clang::CK_FloatingToBoolean:
  case clang::CK_MemberPointerToBoolean:
  case clang::CK_FloatingComplexToBoolean:
  case clang::CK_IntegralComplexToBoolean:
    return true;
  default:
    return false;
  }
}

class ExprVisitor : public clang::RecursiveASTVisitor<ExprVisitor> {
public:
  explicit ExprVisitor(clang::ASTContext *Context) : Context(Context) {}
  bool visitStmt(const clang::Stmt *S, bool &found,
                 clang::QualType &GlobalImplicitCastType) {

    if (isa<clang::ImplicitCastExpr>(S) && !found) {
      const auto CastKind = cast<clang::ImplicitCastExpr>(S)->getCastKind();
      if (isImplicitCastType(CastKind)) {
        found = true;
        const clang::ImplicitCastExpr *ImplicitCast =
            cast<clang::ImplicitCastExpr>(S);
        GlobalImplicitCastType = ImplicitCast->getType();
        // Stop visiting children.
        return false;
      }
    }
    // Continue visiting children.
    for (const clang::Stmt *Child : S->children()) {
      if (Child) {
        this->visitStmt(Child, found, GlobalImplicitCastType);
      }
    }

    return true; // Continue visiting other nodes.
  }

private:
  clang::ASTContext *Context;
};

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

static std::string
createReplacement(const BinaryOperator::Opcode Op, const Expr *CondLhs,
                  const Expr *CondRhs, const Expr *AssignLhs,
                  const ASTContext &Context, const SourceManager &Source,
                  const LangOptions &LO, StringRef FunctionName,
                  QualType GlobalImplicitCastType) {
  const llvm::StringRef CondLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondLhs->getSourceRange()), Source, LO);
  const llvm::StringRef CondRhsStr = Lexer::getSourceText(
      Source.getExpansionRange(CondRhs->getSourceRange()), Source, LO);
  const llvm::StringRef AssignLhsStr = Lexer::getSourceText(
      Source.getExpansionRange(AssignLhs->getSourceRange()), Source, LO);

  return (AssignLhsStr + " = " + FunctionName +
          ((CondLhs->getType()->getUnqualifiedDesugaredType() !=
            CondRhs->getType()->getUnqualifiedDesugaredType())
               ? "<" + GlobalImplicitCastType.getCanonicalType().getAsString() +
                     ">("
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
  bool Found = false;
  clang::QualType GlobalImplicitCastType;

  ExprVisitor Visitor(Result.Context);
  Visitor.visitStmt(If, Found, GlobalImplicitCastType);

  // Ignore Macros
  if (IfLocation.isMacroID() || ThenLocation.isMacroID())
    return;

  // Ignore Dependent types
  if (CondLhs->getType()->isDependentType() ||
      CondRhs->getType()->isDependentType())
    return;

  auto ReplaceAndDiagnose = [&](const llvm::StringRef FunctionName) {
    diag(IfLocation, "use `%0` instead of `%1`")
        << FunctionName << OperatorStr
        << FixItHint::CreateReplacement(
               SourceRange(IfLocation, Lexer::getLocForEndOfToken(
                                           ThenLocation, 0, Source, LO)),
               createReplacement(BinaryOpcode, CondLhs, CondRhs, AssignLhs,
                                 Context, Source, LO, FunctionName,
                                 GlobalImplicitCastType))
        << IncludeInserter.createIncludeInsertion(
               Source.getFileID(If->getBeginLoc()), AlgorithmHeader);
  };

  if (minCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                   Context)) {
    ReplaceAndDiagnose("std::min");
  } else if (maxCondition(BinaryOpcode, CondLhs, CondRhs, AssignLhs, AssignRhs,
                          Context)) {
    ReplaceAndDiagnose("std::max");
  }
}

} // namespace clang::tidy::readability
