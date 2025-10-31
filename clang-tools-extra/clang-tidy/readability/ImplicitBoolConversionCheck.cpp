//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplicitBoolConversionCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include <queue>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(Stmt, isMacroExpansion) {
  SourceManager &SM = Finder->getASTContext().getSourceManager();
  SourceLocation Loc = Node.getBeginLoc();
  return SM.isMacroBodyExpansion(Loc) || SM.isMacroArgExpansion(Loc);
}

AST_MATCHER(Stmt, isC23) { return Finder->getASTContext().getLangOpts().C23; }

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

} // namespace

static StringRef getZeroLiteralToCompareWithForType(CastKind CastExprKind,
                                                    QualType Type,
                                                    ASTContext &Context) {
  switch (CastExprKind) {
  case CK_IntegralToBoolean:
    return Type->isUnsignedIntegerType() ? "0u" : "0";

  case CK_FloatingToBoolean:
    return ASTContext::hasSameType(Type, Context.FloatTy) ? "0.0f" : "0.0";

  case CK_PointerToBoolean:
  case CK_MemberPointerToBoolean: // Fall-through on purpose.
    return (Context.getLangOpts().CPlusPlus11 || Context.getLangOpts().C23)
               ? "nullptr"
               : "0";

  default:
    llvm_unreachable("Unexpected cast kind");
  }
}

static bool isUnaryLogicalNotOperator(const Stmt *Statement) {
  const auto *UnaryOperatorExpr = dyn_cast<UnaryOperator>(Statement);
  return UnaryOperatorExpr && UnaryOperatorExpr->getOpcode() == UO_LNot;
}

static void fixGenericExprCastToBool(DiagnosticBuilder &Diag,
                                     const ImplicitCastExpr *Cast,
                                     const Stmt *Parent, ASTContext &Context,
                                     bool UseUpperCaseLiteralSuffix) {
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

  bool NeedInnerParens =
      utils::fixit::areParensNeededForStatement(*SubExpr->IgnoreImpCasts());
  bool NeedOuterParens =
      Parent != nullptr && utils::fixit::areParensNeededForStatement(*Parent);

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

  const StringRef ZeroLiteral = getZeroLiteralToCompareWithForType(
      Cast->getCastKind(), SubExpr->getType(), Context);

  if (UseUpperCaseLiteralSuffix)
    EndLocInsertion += ZeroLiteral.upper();
  else
    EndLocInsertion += ZeroLiteral;

  if (NeedOuterParens) {
    EndLocInsertion += ")";
  }

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      Cast->getEndLoc(), 0, Context.getSourceManager(), Context.getLangOpts());
  Diag << FixItHint::CreateInsertion(EndLoc, EndLocInsertion);
}

static StringRef getEquivalentBoolLiteralForExpr(const Expr *Expression,
                                                 ASTContext &Context) {
  if (isNULLMacroExpansion(Expression, Context)) {
    return "false";
  }

  if (const auto *IntLit =
          dyn_cast<IntegerLiteral>(Expression->IgnoreParens())) {
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

  return {};
}

static bool needsSpacePrefix(SourceLocation Loc, ASTContext &Context) {
  SourceRange PrefixRange(Loc.getLocWithOffset(-1), Loc);
  StringRef SpaceBeforeStmtStr = Lexer::getSourceText(
      CharSourceRange::getCharRange(PrefixRange), Context.getSourceManager(),
      Context.getLangOpts(), nullptr);
  if (SpaceBeforeStmtStr.empty())
    return true;

  const StringRef AllowedCharacters(" \t\n\v\f\r(){}[]<>;,+=-|&~!^*/");
  return !AllowedCharacters.contains(SpaceBeforeStmtStr.back());
}

static void fixGenericExprCastFromBool(DiagnosticBuilder &Diag,
                                       const ImplicitCastExpr *Cast,
                                       ASTContext &Context,
                                       StringRef OtherType) {
  if (!Context.getLangOpts().CPlusPlus) {
    Diag << FixItHint::CreateInsertion(Cast->getBeginLoc(),
                                       (Twine("(") + OtherType + ")").str());
    return;
  }

  const Expr *SubExpr = Cast->getSubExpr();
  const bool NeedParens = !isa<ParenExpr>(SubExpr->IgnoreImplicit());
  const bool NeedSpace = needsSpacePrefix(Cast->getBeginLoc(), Context);

  Diag << FixItHint::CreateInsertion(
      Cast->getBeginLoc(), (Twine() + (NeedSpace ? " " : "") + "static_cast<" +
                            OtherType + ">" + (NeedParens ? "(" : ""))
                               .str());

  if (NeedParens) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        Cast->getEndLoc(), 0, Context.getSourceManager(),
        Context.getLangOpts());

    Diag << FixItHint::CreateInsertion(EndLoc, ")");
  }
}

static StringRef
getEquivalentForBoolLiteral(const CXXBoolLiteralExpr *BoolLiteral,
                            QualType DestType, ASTContext &Context) {
  // Prior to C++11, false literal could be implicitly converted to pointer.
  if (!Context.getLangOpts().CPlusPlus11 &&
      (DestType->isPointerType() || DestType->isMemberPointerType()) &&
      BoolLiteral->getValue() == false) {
    return "0";
  }

  if (DestType->isFloatingType()) {
    if (ASTContext::hasSameType(DestType, Context.FloatTy)) {
      return BoolLiteral->getValue() ? "1.0f" : "0.0f";
    }
    return BoolLiteral->getValue() ? "1.0" : "0.0";
  }

  if (DestType->isUnsignedIntegerType()) {
    return BoolLiteral->getValue() ? "1u" : "0u";
  }
  return BoolLiteral->getValue() ? "1" : "0";
}

static bool isCastAllowedInCondition(const ImplicitCastExpr *Cast,
                                     ASTContext &Context) {
  std::queue<const Stmt *> Q;
  Q.push(Cast);

  TraversalKindScope RAII(Context, TK_AsIs);

  while (!Q.empty()) {
    for (const auto &N : Context.getParents(*Q.front())) {
      const Stmt *S = N.get<Stmt>();
      if (!S)
        return false;
      if (isa<IfStmt>(S) || isa<ConditionalOperator>(S) || isa<ForStmt>(S) ||
          isa<WhileStmt>(S) || isa<DoStmt>(S) ||
          isa<BinaryConditionalOperator>(S))
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

ImplicitBoolConversionCheck::ImplicitBoolConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowIntegerConditions(Options.get("AllowIntegerConditions", false)),
      AllowPointerConditions(Options.get("AllowPointerConditions", false)),
      UseUpperCaseLiteralSuffix(
          Options.get("UseUpperCaseLiteralSuffix", false)) {}

void ImplicitBoolConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowIntegerConditions", AllowIntegerConditions);
  Options.store(Opts, "AllowPointerConditions", AllowPointerConditions);
  Options.store(Opts, "UseUpperCaseLiteralSuffix", UseUpperCaseLiteralSuffix);
}

void ImplicitBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  auto ExceptionCases =
      expr(anyOf(allOf(isMacroExpansion(), unless(isNULLMacroExpansion())),
                 has(ignoringImplicit(
                     memberExpr(hasDeclaration(fieldDecl(hasBitWidth(1)))))),
                 hasParent(explicitCastExpr()),
                 expr(hasType(qualType().bind("type")),
                      hasParent(initListExpr(hasParent(explicitCastExpr(
                          hasType(qualType(equalsBoundNode("type"))))))))));
  auto ImplicitCastFromBool = implicitCastExpr(
      anyOf(hasCastKind(CK_IntegralCast), hasCastKind(CK_IntegralToFloating),
            // Prior to C++11 cast from bool literal to pointer was allowed.
            allOf(anyOf(hasCastKind(CK_NullToPointer),
                        hasCastKind(CK_NullToMemberPointer)),
                  hasSourceExpression(cxxBoolLiteral()))),
      hasSourceExpression(expr(hasType(booleanType()))));
  auto BoolXor =
      binaryOperator(hasOperatorName("^"), hasLHS(ImplicitCastFromBool),
                     hasRHS(ImplicitCastFromBool));
  auto ComparisonInCall = allOf(
      hasParent(callExpr()),
      hasSourceExpression(binaryOperator(hasAnyOperatorName("==", "!="))));

  auto IsInCompilerGeneratedFunction = hasAncestor(namedDecl(anyOf(
      isImplicit(), functionDecl(isDefaulted()), functionTemplateDecl())));

  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(
                   anyOf(hasCastKind(CK_IntegralToBoolean),
                         hasCastKind(CK_FloatingToBoolean),
                         hasCastKind(CK_PointerToBoolean),
                         hasCastKind(CK_MemberPointerToBoolean)),
                   // Exclude cases of C23 comparison result.
                   unless(allOf(isC23(),
                                hasSourceExpression(ignoringParens(
                                    binaryOperator(hasAnyOperatorName(
                                        ">", ">=", "==", "!=", "<", "<=")))))),
                   // Exclude case of using if or while statements with variable
                   // declaration, e.g.:
                   //   if (int var = functionCall()) {}
                   unless(hasParent(
                       stmt(anyOf(ifStmt(), whileStmt()), has(declStmt())))),
                   // Exclude cases common to implicit cast to and from bool.
                   unless(ExceptionCases), unless(has(BoolXor)),
                   // Exclude C23 cases common to implicit cast to bool.
                   unless(ComparisonInCall),
                   // Retrieve also parent statement, to check if we need
                   // additional parens in replacement.
                   optionally(hasParent(stmt().bind("parentStmt"))),
                   unless(isInTemplateInstantiation()),
                   unless(IsInCompilerGeneratedFunction))
                   .bind("implicitCastToBool")),
      this);

  auto BoolComparison = binaryOperator(hasAnyOperatorName("==", "!="),
                                       hasLHS(ImplicitCastFromBool),
                                       hasRHS(ImplicitCastFromBool));
  auto BoolOpAssignment = binaryOperator(hasAnyOperatorName("|=", "&="),
                                         hasLHS(expr(hasType(booleanType()))));
  auto BitfieldAssignment = binaryOperator(
      hasLHS(memberExpr(hasDeclaration(fieldDecl(hasBitWidth(1))))));
  auto BitfieldConstruct = cxxConstructorDecl(hasDescendant(cxxCtorInitializer(
      withInitializer(equalsBoundNode("implicitCastFromBool")),
      forField(hasBitWidth(1)))));
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          implicitCastExpr(
              ImplicitCastFromBool, unless(ExceptionCases),
              // Exclude comparisons of bools, as they are always cast to
              // integers in such context:
              //   bool_expr_a == bool_expr_b
              //   bool_expr_a != bool_expr_b
              unless(hasParent(
                  binaryOperator(anyOf(BoolComparison, BoolXor,
                                       BoolOpAssignment, BitfieldAssignment)))),
              implicitCastExpr().bind("implicitCastFromBool"),
              unless(hasParent(BitfieldConstruct)),
              // Check also for nested casts, for example: bool -> int -> float.
              optionally(
                  hasParent(implicitCastExpr().bind("furtherImplicitCast"))),
              unless(isInTemplateInstantiation()),
              unless(IsInCompilerGeneratedFunction))),
      this);
}

void ImplicitBoolConversionCheck::check(
    const MatchFinder::MatchResult &Result) {

  if (const auto *CastToBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastToBool")) {
    const auto *Parent = Result.Nodes.getNodeAs<Stmt>("parentStmt");
    handleCastToBool(CastToBool, Parent, *Result.Context);
    return;
  }

  if (const auto *CastFromBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastFromBool")) {
    const auto *NextImplicitCast =
        Result.Nodes.getNodeAs<ImplicitCastExpr>("furtherImplicitCast");
    handleCastFromBool(CastFromBool, NextImplicitCast, *Result.Context);
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

  auto Diag = diag(Cast->getBeginLoc(), "implicit conversion %0 -> 'bool'")
              << Cast->getSubExpr()->getType();

  StringRef EquivalentLiteral =
      getEquivalentBoolLiteralForExpr(Cast->getSubExpr(), Context);
  if (!EquivalentLiteral.empty()) {
    Diag << tooling::fixit::createReplacement(*Cast, EquivalentLiteral);
  } else {
    fixGenericExprCastToBool(Diag, Cast, Parent, Context,
                             UseUpperCaseLiteralSuffix);
  }
}

void ImplicitBoolConversionCheck::handleCastFromBool(
    const ImplicitCastExpr *Cast, const ImplicitCastExpr *NextImplicitCast,
    ASTContext &Context) {
  QualType DestType =
      NextImplicitCast ? NextImplicitCast->getType() : Cast->getType();
  auto Diag = diag(Cast->getBeginLoc(), "implicit conversion 'bool' -> %0")
              << DestType;

  if (const auto *BoolLiteral =
          dyn_cast<CXXBoolLiteralExpr>(Cast->getSubExpr()->IgnoreParens())) {

    const auto EquivalentForBoolLiteral =
        getEquivalentForBoolLiteral(BoolLiteral, DestType, Context);
    if (UseUpperCaseLiteralSuffix)
      Diag << tooling::fixit::createReplacement(
          *Cast, EquivalentForBoolLiteral.upper());
    else
      Diag << tooling::fixit::createReplacement(*Cast,
                                                EquivalentForBoolLiteral);

  } else {
    fixGenericExprCastFromBool(Diag, Cast, Context, DestType.getAsString());
  }
}

} // namespace clang::tidy::readability
