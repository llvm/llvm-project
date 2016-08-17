//===--- ImplicitBoolCastCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ImplicitBoolCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

AST_MATCHER(Stmt, isMacroExpansion) {
  SourceManager &SM = Finder->getASTContext().getSourceManager();
  SourceLocation Loc = Node.getLocStart();
  return SM.isMacroBodyExpansion(Loc) || SM.isMacroArgExpansion(Loc);
}

bool isNULLMacroExpansion(const Stmt *Statement, ASTContext &Context) {
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LO = Context.getLangOpts();
  SourceLocation Loc = Statement->getLocStart();
  return SM.isMacroBodyExpansion(Loc) &&
         clang::Lexer::getImmediateMacroName(Loc, SM, LO) == "NULL";
}

AST_MATCHER(Stmt, isNULLMacroExpansion) {
  return isNULLMacroExpansion(&Node, Finder->getASTContext());
}

ast_matchers::internal::Matcher<Expr> createExceptionCasesMatcher() {
  return expr(anyOf(hasParent(explicitCastExpr()),
                    allOf(isMacroExpansion(), unless(isNULLMacroExpansion())),
                    isInTemplateInstantiation(),
                    hasAncestor(functionTemplateDecl())));
}

StatementMatcher createImplicitCastFromBoolMatcher() {
  return implicitCastExpr(
      unless(createExceptionCasesMatcher()),
      anyOf(hasCastKind(CK_IntegralCast), hasCastKind(CK_IntegralToFloating),
            // Prior to C++11 cast from bool literal to pointer was allowed.
            allOf(anyOf(hasCastKind(CK_NullToPointer),
                        hasCastKind(CK_NullToMemberPointer)),
                  hasSourceExpression(cxxBoolLiteral()))),
      hasSourceExpression(expr(hasType(qualType(booleanType())))));
}

StringRef
getZeroLiteralToCompareWithForGivenType(CastKind CastExpressionKind,
                                        QualType CastSubExpressionType,
                                        ASTContext &Context) {
  switch (CastExpressionKind) {
  case CK_IntegralToBoolean:
    return CastSubExpressionType->isUnsignedIntegerType() ? "0u" : "0";

  case CK_FloatingToBoolean:
    return Context.hasSameType(CastSubExpressionType, Context.FloatTy) ? "0.0f"
                                                                       : "0.0";

  case CK_PointerToBoolean:
  case CK_MemberPointerToBoolean: // Fall-through on purpose.
    return Context.getLangOpts().CPlusPlus11 ? "nullptr" : "0";

  default:
    llvm_unreachable("Unexpected cast kind");
  }
}

bool isUnaryLogicalNotOperator(const Stmt *Statement) {
  const auto *UnaryOperatorExpression =
      llvm::dyn_cast<UnaryOperator>(Statement);
  return UnaryOperatorExpression != nullptr &&
         UnaryOperatorExpression->getOpcode() == UO_LNot;
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
  if (const CXXOperatorCallExpr *OverloadedOperatorCall =
          llvm::dyn_cast<CXXOperatorCallExpr>(Statement)) {
    return areParensNeededForOverloadedOperator(
        OverloadedOperatorCall->getOperator());
  }

  return llvm::isa<BinaryOperator>(Statement) ||
         llvm::isa<UnaryOperator>(Statement);
}

void addFixItHintsForGenericExpressionCastToBool(
    DiagnosticBuilder &Diagnostic, const ImplicitCastExpr *CastExpression,
    const Stmt *ParentStatement, ASTContext &Context) {
  // In case of expressions like (! integer), we should remove the redundant not
  // operator and use inverted comparison (integer == 0).
  bool InvertComparison =
      ParentStatement != nullptr && isUnaryLogicalNotOperator(ParentStatement);
  if (InvertComparison) {
    SourceLocation ParentStartLoc = ParentStatement->getLocStart();
    SourceLocation ParentEndLoc =
        llvm::cast<UnaryOperator>(ParentStatement)->getSubExpr()->getLocStart();
    Diagnostic.AddFixItHint(FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(ParentStartLoc, ParentEndLoc)));

    auto FurtherParents = Context.getParents(*ParentStatement);
    ParentStatement = FurtherParents[0].get<Stmt>();
  }

  const Expr *SubExpression = CastExpression->getSubExpr();

  bool NeedInnerParens = areParensNeededForStatement(SubExpression);
  bool NeedOuterParens = ParentStatement != nullptr &&
                         areParensNeededForStatement(ParentStatement);

  std::string StartLocInsertion;

  if (NeedOuterParens) {
    StartLocInsertion += "(";
  }
  if (NeedInnerParens) {
    StartLocInsertion += "(";
  }

  if (!StartLocInsertion.empty()) {
    SourceLocation StartLoc = CastExpression->getLocStart();
    Diagnostic.AddFixItHint(
        FixItHint::CreateInsertion(StartLoc, StartLocInsertion));
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

  EndLocInsertion += getZeroLiteralToCompareWithForGivenType(
      CastExpression->getCastKind(), SubExpression->getType(), Context);

  if (NeedOuterParens) {
    EndLocInsertion += ")";
  }

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      CastExpression->getLocEnd(), 0, Context.getSourceManager(),
      Context.getLangOpts());
  Diagnostic.AddFixItHint(FixItHint::CreateInsertion(EndLoc, EndLocInsertion));
}

StringRef getEquivalentBoolLiteralForExpression(const Expr *Expression,
                                                ASTContext &Context) {
  if (isNULLMacroExpansion(Expression, Context)) {
    return "false";
  }

  if (const auto *IntLit = llvm::dyn_cast<IntegerLiteral>(Expression)) {
    return (IntLit->getValue() == 0) ? "false" : "true";
  }

  if (const auto *FloatLit = llvm::dyn_cast<FloatingLiteral>(Expression)) {
    llvm::APFloat FloatLitAbsValue = FloatLit->getValue();
    FloatLitAbsValue.clearSign();
    return (FloatLitAbsValue.bitcastToAPInt() == 0) ? "false" : "true";
  }

  if (const auto *CharLit = llvm::dyn_cast<CharacterLiteral>(Expression)) {
    return (CharLit->getValue() == 0) ? "false" : "true";
  }

  if (llvm::isa<StringLiteral>(Expression->IgnoreCasts())) {
    return "true";
  }

  return StringRef();
}

void addFixItHintsForLiteralCastToBool(DiagnosticBuilder &Diagnostic,
                                       const ImplicitCastExpr *CastExpression,
                                       StringRef EquivalentLiteralExpression) {
  SourceLocation StartLoc = CastExpression->getLocStart();
  SourceLocation EndLoc = CastExpression->getLocEnd();

  Diagnostic.AddFixItHint(FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(StartLoc, EndLoc),
      EquivalentLiteralExpression));
}

void addFixItHintsForGenericExpressionCastFromBool(
    DiagnosticBuilder &Diagnostic, const ImplicitCastExpr *CastExpression,
    ASTContext &Context, StringRef OtherType) {
  const Expr *SubExpression = CastExpression->getSubExpr();
  bool NeedParens = !llvm::isa<ParenExpr>(SubExpression);

  std::string StartLocInsertion = "static_cast<";
  StartLocInsertion += OtherType.str();
  StartLocInsertion += ">";
  if (NeedParens) {
    StartLocInsertion += "(";
  }

  SourceLocation StartLoc = CastExpression->getLocStart();
  Diagnostic.AddFixItHint(
      FixItHint::CreateInsertion(StartLoc, StartLocInsertion));

  if (NeedParens) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        CastExpression->getLocEnd(), 0, Context.getSourceManager(),
        Context.getLangOpts());

    Diagnostic.AddFixItHint(FixItHint::CreateInsertion(EndLoc, ")"));
  }
}

StringRef getEquivalentLiteralForBoolLiteral(
    const CXXBoolLiteralExpr *BoolLiteralExpression, QualType DestinationType,
    ASTContext &Context) {
  // Prior to C++11, false literal could be implicitly converted to pointer.
  if (!Context.getLangOpts().CPlusPlus11 &&
      (DestinationType->isPointerType() ||
       DestinationType->isMemberPointerType()) &&
      BoolLiteralExpression->getValue() == false) {
    return "0";
  }

  if (DestinationType->isFloatingType()) {
    if (BoolLiteralExpression->getValue() == true) {
      return Context.hasSameType(DestinationType, Context.FloatTy) ? "1.0f"
                                                                   : "1.0";
    }
    return Context.hasSameType(DestinationType, Context.FloatTy) ? "0.0f"
                                                                 : "0.0";
  }

  if (BoolLiteralExpression->getValue() == true) {
    return DestinationType->isUnsignedIntegerType() ? "1u" : "1";
  }
  return DestinationType->isUnsignedIntegerType() ? "0u" : "0";
}

void addFixItHintsForLiteralCastFromBool(DiagnosticBuilder &Diagnostic,
                                         const ImplicitCastExpr *CastExpression,
                                         ASTContext &Context,
                                         QualType DestinationType) {
  SourceLocation StartLoc = CastExpression->getLocStart();
  SourceLocation EndLoc = CastExpression->getLocEnd();
  const auto *BoolLiteralExpression =
      llvm::dyn_cast<CXXBoolLiteralExpr>(CastExpression->getSubExpr());

  Diagnostic.AddFixItHint(FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(StartLoc, EndLoc),
      getEquivalentLiteralForBoolLiteral(BoolLiteralExpression, DestinationType,
                                         Context)));
}

StatementMatcher createConditionalExpressionMatcher() {
  return stmt(anyOf(ifStmt(), conditionalOperator(),
                    parenExpr(hasParent(conditionalOperator()))));
}

bool isAllowedConditionalCast(const ImplicitCastExpr *CastExpression,
                              ASTContext &Context) {
  auto AllowedConditionalMatcher = stmt(hasParent(stmt(
      anyOf(createConditionalExpressionMatcher(),
            unaryOperator(hasOperatorName("!"),
                          hasParent(createConditionalExpressionMatcher()))))));

  auto MatchResult = match(AllowedConditionalMatcher, *CastExpression, Context);
  return !MatchResult.empty();
}

} // anonymous namespace

ImplicitBoolCastCheck::ImplicitBoolCastCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowConditionalIntegerCasts(
          Options.get("AllowConditionalIntegerCasts", false)),
      AllowConditionalPointerCasts(
          Options.get("AllowConditionalPointerCasts", false)) {}

void ImplicitBoolCastCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowConditionalIntegerCasts",
                AllowConditionalIntegerCasts);
  Options.store(Opts, "AllowConditionalPointerCasts",
                AllowConditionalPointerCasts);
}

void ImplicitBoolCastCheck::registerMatchers(MatchFinder *Finder) {
  // This check doesn't make much sense if we run it on language without
  // built-in bool support.
  if (!getLangOpts().Bool) {
    return;
  }

  Finder->addMatcher(
      implicitCastExpr(
          // Exclude cases common to implicit cast to and from bool.
          unless(createExceptionCasesMatcher()),
          // Exclude case of using if or while statements with variable
          // declaration, e.g.:
          //   if (int var = functionCall()) {}
          unless(
              hasParent(stmt(anyOf(ifStmt(), whileStmt()), has(declStmt())))),
          anyOf(hasCastKind(CK_IntegralToBoolean),
                hasCastKind(CK_FloatingToBoolean),
                hasCastKind(CK_PointerToBoolean),
                hasCastKind(CK_MemberPointerToBoolean)),
          // Retrive also parent statement, to check if we need additional
          // parens in replacement.
          anyOf(hasParent(stmt().bind("parentStmt")), anything()))
          .bind("implicitCastToBool"),
      this);

  Finder->addMatcher(
      implicitCastExpr(
          createImplicitCastFromBoolMatcher(),
          // Exclude comparisons of bools, as they are always cast to integers
          // in such context:
          //   bool_expr_a == bool_expr_b
          //   bool_expr_a != bool_expr_b
          unless(hasParent(binaryOperator(
              anyOf(hasOperatorName("=="), hasOperatorName("!=")),
              hasLHS(createImplicitCastFromBoolMatcher()),
              hasRHS(createImplicitCastFromBoolMatcher())))),
          // Check also for nested casts, for example: bool -> int -> float.
          anyOf(hasParent(implicitCastExpr().bind("furtherImplicitCast")),
                anything()))
          .bind("implicitCastFromBool"),
      this);
}

void ImplicitBoolCastCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *CastToBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastToBool")) {
    const auto *ParentStatement = Result.Nodes.getNodeAs<Stmt>("parentStmt");
    return handleCastToBool(CastToBool, ParentStatement, *Result.Context);
  }

  if (const auto *CastFromBool =
          Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCastFromBool")) {
    const auto *FurtherImplicitCastExpression =
        Result.Nodes.getNodeAs<ImplicitCastExpr>("furtherImplicitCast");
    return handleCastFromBool(CastFromBool, FurtherImplicitCastExpression,
                              *Result.Context);
  }
}

void ImplicitBoolCastCheck::handleCastToBool(
    const ImplicitCastExpr *CastExpression, const Stmt *ParentStatement,
    ASTContext &Context) {
  if (AllowConditionalPointerCasts &&
      (CastExpression->getCastKind() == CK_PointerToBoolean ||
       CastExpression->getCastKind() == CK_MemberPointerToBoolean) &&
      isAllowedConditionalCast(CastExpression, Context)) {
    return;
  }

  if (AllowConditionalIntegerCasts &&
      CastExpression->getCastKind() == CK_IntegralToBoolean &&
      isAllowedConditionalCast(CastExpression, Context)) {
    return;
  }

  std::string OtherType = CastExpression->getSubExpr()->getType().getAsString();
  DiagnosticBuilder Diagnostic =
      diag(CastExpression->getLocStart(), "implicit cast '%0' -> bool")
      << OtherType;

  StringRef EquivalentLiteralExpression = getEquivalentBoolLiteralForExpression(
      CastExpression->getSubExpr(), Context);
  if (!EquivalentLiteralExpression.empty()) {
    addFixItHintsForLiteralCastToBool(Diagnostic, CastExpression,
                                      EquivalentLiteralExpression);
  } else {
    addFixItHintsForGenericExpressionCastToBool(Diagnostic, CastExpression,
                                                ParentStatement, Context);
  }
}

void ImplicitBoolCastCheck::handleCastFromBool(
    const ImplicitCastExpr *CastExpression,
    const ImplicitCastExpr *FurtherImplicitCastExpression,
    ASTContext &Context) {
  QualType DestinationType = (FurtherImplicitCastExpression != nullptr)
                                 ? FurtherImplicitCastExpression->getType()
                                 : CastExpression->getType();
  std::string DestinationTypeString = DestinationType.getAsString();
  DiagnosticBuilder Diagnostic =
      diag(CastExpression->getLocStart(), "implicit cast bool -> '%0'")
      << DestinationTypeString;

  if (llvm::isa<CXXBoolLiteralExpr>(CastExpression->getSubExpr())) {
    addFixItHintsForLiteralCastFromBool(Diagnostic, CastExpression, Context,
                                        DestinationType);
  } else {
    addFixItHintsForGenericExpressionCastFromBool(
        Diagnostic, CastExpression, Context, DestinationTypeString);
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
