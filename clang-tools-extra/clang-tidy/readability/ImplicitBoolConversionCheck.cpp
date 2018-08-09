//===--- ImplicitBoolConversionCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ImplicitBoolConversionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include <queue>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

AST_MATCHER(Stmt, isMacroExpansion) {
  SourceManager &SM = Finder->getASTContext().getSourceManager();
  SourceLocation Loc = Node.getBeginLoc();
  return SM.isMacroBodyExpansion(Loc) || SM.isMacroArgExpansion(Loc);
}

bool isNULLMacroExpansion(const Stmt *Statement, ASTContext &Context) {
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LO = Context.getLangOpts();
  SourceLocation Loc = Statement->getBeginLoc();
  return SM.isMacroBodyExpansion(Loc) &&
         Lexer::getImmediateMacroName(Loc, SM, LO) == "NULL";
}

AST_MATCHER(Stmt, isNULLMacroExpansion) {
  return isNULLMacroExpansion(&Node, Finder->getASTContext());
}

StringRef getZeroLiteralToCompareWithForType(CastKind CastExprKind,
                                             QualType Type,
                                             ASTContext &Context) {
  switch (CastExprKind) {
  case CK_IntegralToBoolean:
    return Type->isUnsignedIntegerType() ? "0u" : "0";

  case CK_FloatingToBoolean:
    return Context.hasSameType(Type, Context.FloatTy) ? "0.0f" : "0.0";

  case CK_PointerToBoolean:
  case CK_MemberPointerToBoolean: // Fall-through on purpose.
    return Context.getLangOpts().CPlusPlus11 ? "nullptr" : "0";

  default:
    llvm_unreachable("Unexpected cast kind");
  }
}

bool isUnaryLogicalNotOperator(const Stmt *Statement) {
  const auto *UnaryOperatorExpr = dyn_cast<UnaryOperator>(Statement);
  return UnaryOperatorExpr && UnaryOperatorExpr->getOpcode() == UO_LNot;
}

bool areParensNeededForOverloadedOperator(OverloadedOperatorKind OperatorKind) {
  switch (OperatorKind) {
  case OO_New:
  case OO_Delete: // Fall-through on purpose.
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_ArrowStar:
  case OO_Arrow:
  case OO_Call:
  case OO_Subscript:
    return false;

  default:
    return true;
  }
}

bool areParensNeededForStatement(const Stmt *Statement) {
  if (const auto *OperatorCall = dyn_cast<CXXOperatorCallExpr>(Statement)) {
    return areParensNeededForOverloadedOperator(OperatorCall->getOperator());
  }

  return isa<BinaryOperator>(Statement) || isa<UnaryOperator>(Statement);
}

void fixGenericExprCastToBool(DiagnosticBuilder &Diag,
                              const ImplicitCastExpr *Cast, const Stmt *Parent,
                              ASTContext &Context) {
  // In case of expressions like (! integer), we should remove the redundant not
  // operator and use inverted comparison (integer == 0).
  bool InvertComparison =
      Parent != nullptr && isUnaryLogicalNotOperator(Parent);
  if (InvertComparison) {
    SourceLocation ParentStartLoc = Parent->getBeginLoc();
    SourceLocation ParentEndLoc =
        cast<UnaryOperator>(Parent)->getSubExpr()->getBeginLoc();
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(ParentStartLoc, ParentEndLoc));

    Parent = Context.getParents(*Parent)[0].get<Stmt>();
  }

  const Expr *SubExpr = Cast->getSubExpr();

  bool NeedInnerParens = areParensNeededForStatement(SubExpr);
  bool NeedOuterParens =
      Parent != nullptr && areParensNeededForStatement(Parent);

  std::string StartLocInsertion;

  if (NeedOuterParens) {
    StartLocInsertion += "(";
  }
  if (NeedInnerParens) {
    StartLocInsertion += "(";
  }

  if (!StartLocInsertion.empty()) {
    Diag << FixItHint::CreateInsertion(Cast->getBeginLoc(), StartLocInsertion);
  }

  std::string EndLocInsertion;

  if (NeedInnerParens) {
    EndLocInsertion += ")";
  }

  if (InvertComparison) {
    EndLocInsertion += " == ";
  } else {
    EndLocInsertion += " != ";
  }

  EndLocInsertion += getZeroLiteralToCompareWithForType(
      Cast->getCastKind(), SubExpr->getType(), Context);

  if (NeedOuterParens) {
    EndLocInsertion += ")";
  }

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      Cast->getEndLoc(), 0, Context.getSourceManager(), Context.getLangOpts());
  Diag << FixItHint::CreateInsertion(EndLoc, EndLocInsertion);
}

StringRef getEquivalentBoolLiteralForExpr(const Expr *Expression,
                                          ASTContext &Context) {
  if (isNULLMacroExpansion(Expression, Context)) {
    return "false";
  }

  if (const auto *IntLit = dyn_cast<IntegerLiteral>(Expression)) {
    return (IntLit->getValue() == 0) ? "false" : "true";
  }

  if (const auto *FloatLit = dyn_cast<FloatingLiteral>(Expression)) {
    llvm::APFloat FloatLitAbsValue = FloatLit->getValue();
    FloatLitAbsValue.clearSign();
    return (FloatLitAbsValue.bitcastToAPInt() == 0) ? "false" : "true";
  }

  if (const auto *CharLit = dyn_cast<CharacterLiteral>(Expression)) {
    return (CharLit->getValue() == 0) ? "false" : "true";
  }

  if (isa<StringLiteral>(Expression->IgnoreCasts())) {
    return "true";
  }

  return StringRef();
}

void fixGenericExprCastFromBool(DiagnosticBuilder &Diag,
                                const ImplicitCastExpr *Cast,
                                ASTContext &Context, StringRef OtherType) {
  const Expr *SubExpr = Cast->getSubExpr();
  bool NeedParens = !isa<ParenExpr>(SubExpr);

  Diag << FixItHint::CreateInsertion(
      Cast->getBeginLoc(),
      (Twine("static_cast<") + OtherType + ">" + (NeedParens ? "(" : ""))
          .str());

  if (NeedParens) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        Cast->getEndLoc(), 0, Context.getSourceManager(),
        Context.getLangOpts());

    Diag << FixItHint::CreateInsertion(EndLoc, ")");
  }
}

StringRef getEquivalentForBoolLiteral(const CXXBoolLiteralExpr *BoolLiteral,
                                      QualType DestType, ASTContext &Context) {
  // Prior to C++11, false literal could be implicitly converted to pointer.
  if (!Context.getLangOpts().CPlusPlus11 &&
      (DestType->isPointerType() || DestType->isMemberPointerType()) &&
      BoolLiteral->getValue() == false) {
    return "0";
  }

  if (DestType->isFloatingType()) {
    if (Context.hasSameType(DestType, Context.FloatTy)) {
      return BoolLiteral->getValue() ? "1.0f" : "0.0f";
    }
    return BoolLiteral->getValue() ? "1.0" : "0.0";
  }

  if (DestType->isUnsignedIntegerType()) {
    return BoolLiteral->getValue() ? "1u" : "0u";
  }
  return BoolLiteral->getValue() ? "1" : "0";
}

bool isCastAllowedInCondition(const ImplicitCastExpr *Cast,
                              ASTContext &Context) {
  std::queue<const Stmt *> Q;
  Q.push(Cast);
  while (!Q.empty()) {
    for (const auto &N : Context.getParents(*Q.front())) {
      const Stmt *S = N.get<Stmt>();
      if (!S)
        return false;
      if (isa<IfStmt>(S) || isa<ConditionalOperator>(S) || isa<ForStmt>(S) ||
          isa<WhileStmt>(S) || isa<BinaryConditionalOperator>(S))
        return true;
      if (isa<ParenExpr>(S) || isa<ImplicitCastExpr>(S) ||
          isUnaryLogicalNotOperator(S) ||
          (isa<BinaryOperator>(S) && cast<BinaryOperator>(S)->isLogicalOp())) {
        Q.push(S);
      } else {
        return false;
      }
    }
    Q.pop();
  }
  return false;
}

} // anonymous namespace

ImplicitBoolConversionCheck::ImplicitBoolConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowIntegerConditions(Options.get("AllowIntegerConditions", false)),
      AllowPointerConditions(Options.get("AllowPointerConditions", false)) {}

void ImplicitBoolConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowIntegerConditions", AllowIntegerConditions);
  Options.store(Opts, "AllowPointerConditions", AllowPointerConditions);
}

void ImplicitBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  // This check doesn't make much sense if we run it on language without
  // built-in bool support.
  if (!getLangOpts().Bool) {
    return;
  }

  auto exceptionCases =
      expr(anyOf(allOf(isMacroExpansion(), unless(isNULLMacroExpansion())),
                 hasParent(explicitCastExpr())));
  auto implicitCastFromBool = implicitCastExpr(
      anyOf(hasCastKind(CK_IntegralCast), hasCastKind(CK_IntegralToFloating),
            // Prior to C++11 cast from bool literal to pointer was allowed.
            allOf(anyOf(hasCastKind(CK_NullToPointer),
                        hasCastKind(CK_NullToMemberPointer)),
                  hasSourceExpression(cxxBoolLiteral()))),
      hasSourceExpression(expr(hasType(booleanType()))),
      unless(exceptionCases));
  auto boolXor =
      binaryOperator(hasOperatorName("^"), hasLHS(implicitCastFromBool),
                     hasRHS(implicitCastFromBool));
  Finder->addMatcher(
      implicitCastExpr(
          anyOf(hasCastKind(CK_IntegralToBoolean),
                hasCastKind(CK_FloatingToBoolean),
                hasCastKind(CK_PointerToBoolean),
                hasCastKind(CK_MemberPointerToBoolean)),
          // Exclude case of using if or while statements with variable
          // declaration, e.g.:
          //   if (int var = functionCall()) {}
          unless(
              hasParent(stmt(anyOf(ifStmt(), whileStmt()), has(declStmt())))),
          // Exclude cases common to implicit cast to and from bool.
          unless(exceptionCases), unless(has(boolXor)),
          // Retrive also parent statement, to check if we need additional
          // parens in replacement.
          anyOf(hasParent(stmt().bind("parentStmt")), anything()),
          unless(isInTemplateInstantiation()),
          unless(hasAncestor(functionTemplateDecl())))
          .bind("implicitCastToBool"),
      this);

  auto boolComparison = binaryOperator(
      anyOf(hasOperatorName("=="), hasOperatorName("!=")),
      hasLHS(implicitCastFromBool), hasRHS(implicitCastFromBool));
  auto boolOpAssignment =
      binaryOperator(anyOf(hasOperatorName("|="), hasOperatorName("&=")),
                     hasLHS(expr(hasType(booleanType()))));
  Finder->addMatcher(
      implicitCastExpr(
          implicitCastFromBool,
          // Exclude comparisons of bools, as they are always cast to integers
          // in such context:
          //   bool_expr_a == bool_expr_b
          //   bool_expr_a != bool_expr_b
          unless(hasParent(binaryOperator(
              anyOf(boolComparison, boolXor, boolOpAssignment)))),
          // Check also for nested casts, for example: bool -> int -> float.
          anyOf(hasParent(implicitCastExpr().bind("furtherImplicitCast")),
                anything()),
          unless(isInTemplateInstantiation()),
          unless(hasAncestor(functionTemplateDecl())))
          .bind("implicitCastFromBool"),
      this);
}

void ImplicitBoolConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *CastToBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastToBool")) {
    const auto *Parent = Result.Nodes.getNodeAs<Stmt>("parentStmt");
    return handleCastToBool(CastToBool, Parent, *Result.Context);
  }

  if (const auto *CastFromBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastFromBool")) {
    const auto *NextImplicitCast =
        Result.Nodes.getNodeAs<ImplicitCastExpr>("furtherImplicitCast");
    return handleCastFromBool(CastFromBool, NextImplicitCast, *Result.Context);
  }
}

void ImplicitBoolConversionCheck::handleCastToBool(const ImplicitCastExpr *Cast,
                                                   const Stmt *Parent,
                                                   ASTContext &Context) {
  if (AllowPointerConditions &&
      (Cast->getCastKind() == CK_PointerToBoolean ||
       Cast->getCastKind() == CK_MemberPointerToBoolean) &&
      isCastAllowedInCondition(Cast, Context)) {
    return;
  }

  if (AllowIntegerConditions && Cast->getCastKind() == CK_IntegralToBoolean &&
      isCastAllowedInCondition(Cast, Context)) {
    return;
  }

  auto Diag = diag(Cast->getBeginLoc(), "implicit conversion %0 -> bool")
              << Cast->getSubExpr()->getType();

  StringRef EquivalentLiteral =
      getEquivalentBoolLiteralForExpr(Cast->getSubExpr(), Context);
  if (!EquivalentLiteral.empty()) {
    Diag << tooling::fixit::createReplacement(*Cast, EquivalentLiteral);
  } else {
    fixGenericExprCastToBool(Diag, Cast, Parent, Context);
  }
}

void ImplicitBoolConversionCheck::handleCastFromBool(
    const ImplicitCastExpr *Cast, const ImplicitCastExpr *NextImplicitCast,
    ASTContext &Context) {
  QualType DestType =
      NextImplicitCast ? NextImplicitCast->getType() : Cast->getType();
  auto Diag = diag(Cast->getBeginLoc(), "implicit conversion bool -> %0")
              << DestType;

  if (const auto *BoolLiteral =
          dyn_cast<CXXBoolLiteralExpr>(Cast->getSubExpr())) {
    Diag << tooling::fixit::createReplacement(
        *Cast, getEquivalentForBoolLiteral(BoolLiteral, DestType, Context));
  } else {
    fixGenericExprCastFromBool(Diag, Cast, Context, DestType.getAsString());
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
