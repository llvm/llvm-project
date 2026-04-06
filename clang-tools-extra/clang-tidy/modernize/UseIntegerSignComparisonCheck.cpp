//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseIntegerSignComparisonCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::modernize {

/// Find if the passed type is the actual "char" type,
/// not applicable to explicit "signed char" or "unsigned char" types.
static bool isActualCharType(const QualType &Ty) {
  using namespace clang;
  const Type *DesugaredType = Ty->getUnqualifiedDesugaredType();
  if (const auto *BT = dyn_cast<BuiltinType>(DesugaredType))
    return (BT->getKind() == BuiltinType::Char_U ||
            BT->getKind() == BuiltinType::Char_S);
  return false;
}

namespace {
AST_MATCHER(QualType, isActualChar) { return isActualCharType(Node); }
AST_MATCHER(Expr, hasSideEffects) {
  return Node.HasSideEffects(Finder->getASTContext());
}
} // namespace

static BindableMatcher<Stmt> intCastExpression(bool IsSigned,
                                               StringRef CastBindName = {}) {
  // std::cmp_{} functions trigger a compile-time error if either LHS or RHS
  // is a non-integer type, char, enum or bool
  // (unsigned char/ signed char are Ok and can be used).
  auto IntTypeExpr = expr(hasType(hasCanonicalType(qualType(
      IsSigned ? isSignedInteger() : isUnsignedInteger(),
      unless(isActualChar()), unless(booleanType()), unless(enumType())))));

  const auto ImplicitCastExpr =
      CastBindName.empty() ? implicitCastExpr(hasSourceExpression(IntTypeExpr))
                           : implicitCastExpr(hasSourceExpression(IntTypeExpr))
                                 .bind(CastBindName);

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  return expr(anyOf(ImplicitCastExpr, CStyleCastExpr, StaticCastExpr,
                    FunctionalCastExpr));
}

/// Extract the source text of the first template argument from a
/// numeric_limits<T>::min/max/lowest() call expression, preserving typedef
/// aliases as written (e.g. int32_t rather than the canonical int).
static StringRef getLimitsTypeSourceText(const CallExpr *CE,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  const Expr *Callee = CE->getCallee()->IgnoreImplicit();
  NestedNameSpecifierLoc QualLoc;
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Callee))
    QualLoc = DRE->getQualifierLoc();
  else if (const auto *ME = dyn_cast<MemberExpr>(Callee))
    QualLoc = ME->getQualifierLoc();
  if (!QualLoc.getNestedNameSpecifier())
    return {};
  TypeLoc TL = QualLoc.getAsTypeLoc();
  if (TL.isNull())
    return {};
  auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>();
  if (TSTL.isNull() || TSTL.getNumArgs() == 0)
    return {};
  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(TSTL.getArgLoc(0).getSourceRange()),
      SM, LangOpts);
}

static StringRef parseOpCode(BinaryOperator::Opcode Code) {
  switch (Code) {
  case BO_LT:
    return "cmp_less";
  case BO_GT:
    return "cmp_greater";
  case BO_LE:
    return "cmp_less_equal";
  case BO_GE:
    return "cmp_greater_equal";
  case BO_EQ:
    return "cmp_equal";
  case BO_NE:
    return "cmp_not_equal";
  default:
    llvm_unreachable("invalid opcode");
  }
}

UseIntegerSignComparisonCheck::UseIntegerSignComparisonCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      EnableQtSupport(Options.get("EnableQtSupport", false)) {}

void UseIntegerSignComparisonCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "EnableQtSupport", EnableQtSupport);
}

void UseIntegerSignComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntCastExpr = intCastExpression(true, "sIntCastExpression");
  const auto UnSignedIntCastExpr = intCastExpression(false);

  // Flag all operators "==", "<=", ">=", "<", ">", "!="
  // that are used between signed/unsigned integers.  Range-check
  // sub-comparisons (e.g. val >= limits::min()) are filtered in
  // onEndOfTranslationUnit() after the in_range matches are collected.
  const auto CompareOperator =
      binaryOperator(hasAnyOperatorName("==", "<=", ">=", "<", ">", "!="),
                     hasOperands(SignedIntCastExpr, UnSignedIntCastExpr),
                     unless(isInTemplateInstantiation()))
          .bind("intComparison");

  Finder->addMatcher(CompareOperator, this);

  // Match manual integer range checks using std::numeric_limits that can be
  // replaced with std::in_range / q20::in_range.
  auto IntType = qualType(hasCanonicalType(qualType(
      anyOf(isSignedInteger(), isUnsignedInteger()), unless(isActualChar()),
      unless(booleanType()), unless(enumType()))));

  // Matches "A op B" or the equivalent "B rev_op A", capturing oriented
  // comparisons without listing every commutative variant separately.
  const auto Ordered = [](const char *Op, const char *RevOp, const auto &A,
                           const auto &B) {
    return binaryOperator(
        anyOf(allOf(hasOperatorName(Op), hasLHS(ignoringParenImpCasts(A)),
                    hasRHS(ignoringParenImpCasts(B))),
              allOf(hasOperatorName(RevOp), hasLHS(ignoringParenImpCasts(B)),
                    hasRHS(ignoringParenImpCasts(A)))));
  };

  // Matches std::numeric_limits<T>::<Names>(), binding the template arg type.
  const auto LimitsCall = [&](auto Names, StringRef TypeBind) {
    return callExpr(
        argumentCountIs(0),
        callee(cxxMethodDecl(
            Names, ofClass(classTemplateSpecializationDecl(
                       hasName("numeric_limits"), isInStdNamespace(),
                       hasTemplateArgument(
                           0, refersToType(IntType.bind(TypeBind))))))));
  };

  auto LimitsMin =
      LimitsCall(hasAnyName("min", "lowest"), "MinType").bind("LimitsMinExpr");
  auto LimitsMax = LimitsCall(hasName("max"), "MaxType");
  auto ValueLower =
      expr(hasType(IntType), unless(hasSideEffects())).bind("ValueFromLower");
  auto ValueUpper =
      expr(hasType(IntType), unless(hasSideEffects())).bind("ValueFromUpper");

  // Form 1: val >= min() && val <= max() (and commutative/swapped variants)
  Finder->addMatcher(
      binaryOperator(hasOperatorName("&&"),
                     hasOperands(ignoringParenImpCasts(Ordered(">=", "<=", ValueLower, LimitsMin)),
                                 ignoringParenImpCasts(Ordered("<=", ">=", ValueUpper, LimitsMax))),
                     unless(isInTemplateInstantiation()))
          .bind("RangeCheck"),
      this);

  // Form 2: !(val < min() || val > max()) (and commutative/swapped variants)
  Finder->addMatcher(
      unaryOperator(
          hasOperatorName("!"),
          hasUnaryOperand(ignoringParenImpCasts(binaryOperator(
              hasOperatorName("||"),
              hasOperands(ignoringParenImpCasts(Ordered("<", ">", ValueLower, LimitsMin)),
                          ignoringParenImpCasts(Ordered(">", "<", ValueUpper, LimitsMax)))))),
          unless(isInTemplateInstantiation()))
          .bind("RangeCheck"),
      this);
}

void UseIntegerSignComparisonCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseIntegerSignComparisonCheck::check(
    const MatchFinder::MatchResult &Result) {
  StringRef CmpNamespace;
  StringRef CmpHeader;
  if (getLangOpts().CPlusPlus20) {
    CmpHeader = "<utility>";
    CmpNamespace = "std::";
  } else if (getLangOpts().CPlusPlus17 && EnableQtSupport) {
    CmpHeader = "<QtCore/q20utility.h>";
    CmpNamespace = "q20::";
  }

  // Handle in_range pattern (val >= min && val <= max, or negated form).
  if (const auto *Matched = Result.Nodes.getNodeAs<Expr>("RangeCheck")) {
    const auto *ValueLower = Result.Nodes.getNodeAs<Expr>("ValueFromLower");
    const auto *ValueUpper = Result.Nodes.getNodeAs<Expr>("ValueFromUpper");
    const auto *MinTypePtr = Result.Nodes.getNodeAs<QualType>("MinType");
    const auto *MaxTypePtr = Result.Nodes.getNodeAs<QualType>("MaxType");
    if (!ValueLower || !ValueUpper || !MinTypePtr || !MaxTypePtr)
      return;

    if (Result.Context->getCanonicalType(*MinTypePtr) !=
        Result.Context->getCanonicalType(*MaxTypePtr))
      return;

    if (!utils::areStatementsIdentical(ValueLower, ValueUpper, *Result.Context,
                                       /*Canonical=*/true))
      return;

    const SourceManager &SM = *Result.SourceManager;
    StringRef ValueText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(SM.getSpellingLoc(ValueLower->getBeginLoc()),
                                       SM.getSpellingLoc(ValueLower->getEndLoc())),
        SM, getLangOpts());
    if (ValueText.empty())
      return;

    std::string TypeStr;
    if (const auto *MinCall =
            Result.Nodes.getNodeAs<CallExpr>("LimitsMinExpr")) {
      StringRef Written = getLimitsTypeSourceText(MinCall, SM, getLangOpts());
      if (!Written.empty())
        TypeStr = Written.str();
    }
    if (TypeStr.empty())
      TypeStr = MinTypePtr->getAsString(Result.Context->getPrintingPolicy());

    // Record the source range so onEndOfTranslationUnit() can suppress any
    // sign-comparison diagnostics for the sub-comparisons.
    SrcMgr = Result.SourceManager;
    RangeCheckRanges.push_back(Matched->getSourceRange());

    auto Diag = diag(Matched->getBeginLoc(),
                     "use '%0in_range' instead of manual range check")
                << CmpNamespace;
    if (!Matched->getBeginLoc().isMacroID() && !Matched->getEndLoc().isMacroID()) {
      std::string Replacement =
          (CmpNamespace + "in_range<" + TypeStr + ">(" + ValueText + ")").str();
      Diag << FixItHint::CreateReplacement(Matched->getSourceRange(), Replacement)
           << IncludeInserter.createIncludeInsertion(
                  SM.getFileID(Matched->getBeginLoc()), CmpHeader);
    }
    return;
  }

  const auto *SignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("sIntCastExpression");
  assert(SignedCastExpression);

  // Ignore the match if we know that the signed int value is not negative.
  Expr::EvalResult EVResult;
  if (!SignedCastExpression->isValueDependent() &&
      SignedCastExpression->getSubExpr()->EvaluateAsInt(EVResult,
                                                        *Result.Context) &&
      EVResult.Val.getInt().isNonNegative())
    return;

  const auto *BinaryOp =
      Result.Nodes.getNodeAs<BinaryOperator>("intComparison");
  assert(BinaryOp);

  SrcMgr = Result.SourceManager;
  PendingCmps.push_back({BinaryOp});
}

void UseIntegerSignComparisonCheck::onEndOfTranslationUnit() {
  if (PendingCmps.empty()) {
    RangeCheckRanges.clear();
    return;
  }

  StringRef CmpNamespace;
  StringRef CmpHeader;
  if (getLangOpts().CPlusPlus20) {
    CmpHeader = "<utility>";
    CmpNamespace = "std::";
  } else if (getLangOpts().CPlusPlus17 && EnableQtSupport) {
    CmpHeader = "<QtCore/q20utility.h>";
    CmpNamespace = "q20::";
  }

  for (const auto &Pending : PendingCmps) {
    const auto *BinaryOp = Pending.BinaryOp;
    SourceLocation BOpLoc = BinaryOp->getBeginLoc();

    // Skip sub-comparisons that are part of a recognized range check; those
    // are already covered by an in_range diagnostic.
    bool InRangeCheck =
        llvm::any_of(RangeCheckRanges, [&](const SourceRange &RR) {
          return !SrcMgr->isBeforeInTranslationUnit(BOpLoc, RR.getBegin()) &&
                 !SrcMgr->isBeforeInTranslationUnit(RR.getEnd(), BOpLoc);
        });
    if (InRangeCheck)
      continue;

    const Expr *LHS = BinaryOp->getLHS()->IgnoreImpCasts();
    const Expr *RHS = BinaryOp->getRHS()->IgnoreImpCasts();
    const Expr *SubExprLHS = nullptr;
    const Expr *SubExprRHS = nullptr;
    SourceRange R1(LHS->getBeginLoc());
    SourceRange R2(BinaryOp->getOperatorLoc());
    SourceRange R3(Lexer::getLocForEndOfToken(
        RHS->getEndLoc(), 0, *SrcMgr, getLangOpts()));
    if (const auto *LHSCast = dyn_cast<ExplicitCastExpr>(LHS)) {
      SubExprLHS = LHSCast->getSubExpr();
      R1.setEnd(SubExprLHS->getBeginLoc().getLocWithOffset(-1));
      R2.setBegin(Lexer::getLocForEndOfToken(SubExprLHS->getEndLoc(), 0,
                                              *SrcMgr, getLangOpts()));
    }
    if (const auto *RHSCast = dyn_cast<ExplicitCastExpr>(RHS)) {
      SubExprRHS = RHSCast->getSubExpr();
      R2.setEnd(SubExprRHS->getBeginLoc().getLocWithOffset(-1));
      R3.setBegin(Lexer::getLocForEndOfToken(SubExprRHS->getEndLoc(), 0,
                                              *SrcMgr, getLangOpts()));
    }
    const DiagnosticBuilder Diag =
        diag(BinaryOp->getBeginLoc(),
             "comparison between 'signed' and 'unsigned' integers");
    Diag << FixItHint::CreateReplacement(
        CharSourceRange(R1, SubExprLHS != nullptr),
        Twine(CmpNamespace + parseOpCode(BinaryOp->getOpcode()) + "(").str());
    Diag << FixItHint::CreateReplacement(R2, ",");
    Diag << FixItHint::CreateReplacement(CharSourceRange::getCharRange(R3), ")");
    Diag << IncludeInserter.createIncludeInsertion(
        SrcMgr->getFileID(BinaryOp->getBeginLoc()), CmpHeader);
  }

  PendingCmps.clear();
  RangeCheckRanges.clear();
  SrcMgr = nullptr;
}

} // namespace clang::tidy::modernize
