//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientAlgorithmCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static bool areTypesCompatible(QualType Left, QualType Right) {
  if (const auto *LeftRefType = Left->getAs<ReferenceType>())
    Left = LeftRefType->getPointeeType();
  if (const auto *RightRefType = Right->getAs<ReferenceType>())
    Right = RightRefType->getPointeeType();
  return Left->getCanonicalTypeUnqualified() ==
         Right->getCanonicalTypeUnqualified();
}

/// Returns true if built-in `<` orders `T` the way `std::less<T>` does,
/// excluding floating point (NaN compares unordered).
static bool hasBuiltinOrder(QualType T) {
  return T->isIntegralOrEnumerationType() || T->isPointerType();
}

/// Returns true if converting `Bound` to `KeyType` cannot change its value.
static bool boundConvertsExactly(const Expr *Bound, QualType KeyType,
                                 const ASTContext &Ctx) {
  if (!KeyType->isIntegralType(Ctx))
    return false;
  const bool KeySigned = KeyType->isSignedIntegerOrEnumerationType();

  const QualType BoundType = Bound->getType();
  if (BoundType->isIntegerType()) {
    const bool BoundSigned = BoundType->isSignedIntegerOrEnumerationType();
    const unsigned BoundWidth = Ctx.getIntWidth(BoundType);
    const unsigned KeyWidth = Ctx.getIntWidth(KeyType);
    if ((BoundSigned == KeySigned) ? (BoundWidth <= KeyWidth)
                                   : (!BoundSigned && BoundWidth < KeyWidth))
      return true;
  }

  // Otherwise the bound must be a constant that survives the conversion.
  Expr::EvalResult Eval;
  if (!Bound->EvaluateAsInt(Eval, Ctx))
    return false;
  const llvm::APSInt Value = Eval.Val.getInt();
  llvm::APSInt Converted = Value.extOrTrunc(Ctx.getIntWidth(KeyType));
  Converted.setIsSigned(KeySigned);
  return llvm::APSInt::isSameValue(Converted, Value);
}

/// Matches a call taking `c.begin()` and `c.end()` as its first two arguments,
/// where `c` is a `Container`. Binds `c` as "IneffContExpr", its
/// declaration as "IneffContObj" and its class as "IneffCont", or as
/// "IneffContPtr" when `c` is a pointer.
static ast_matchers::internal::Matcher<CallExpr> hasWholeContainerRange(
    const ast_matchers::internal::BindableMatcher<Decl> &Container) {
  return callExpr(
      hasArgument(
          0, cxxMemberCallExpr(
                 callee(cxxMethodDecl(hasName("begin"))),
                 on(declRefExpr(hasDeclaration(decl().bind("IneffContObj")),
                                anyOf(hasType(Container.bind("IneffCont")),
                                      hasType(pointsTo(
                                          Container.bind("IneffContPtr")))))
                        .bind("IneffContExpr")))),
      hasArgument(1,
                  cxxMemberCallExpr(callee(cxxMethodDecl(hasName("end"))),
                                    on(declRefExpr(hasDeclaration(
                                        equalsBoundNode("IneffContObj")))))));
}

void InefficientAlgorithmCheck::registerMatchers(MatchFinder *Finder) {
  const auto Algorithms =
      hasAnyName("::std::find", "::std::count", "::std::equal_range",
                 "::std::lower_bound", "::std::upper_bound");
  const auto ContainerMatcher = classTemplateSpecializationDecl(hasAnyName(
      "::std::set", "::std::map", "::std::multiset", "::std::multimap",
      "::std::unordered_set", "::std::unordered_map",
      "::std::unordered_multiset", "::std::unordered_multimap"));

  Finder->addMatcher(callExpr(callee(functionDecl(Algorithms)),
                              argumentCountAtLeast(3),
                              hasWholeContainerRange(ContainerMatcher))
                         .bind("IneffAlg"),
                     this);

  // `upper_bound` (for `>`) and `lower_bound` (for `>=`) binary search where
  // `std::find_if` linearly scans. Only in containers ordered by `std::less`.
  const auto SortedContainer = classTemplateSpecializationDecl(
      hasAnyName("::std::set", "::std::multiset"),
      hasTemplateArgument(
          1, refersToType(hasDeclaration(
                 classTemplateSpecializationDecl(hasName("::std::less"))))));
  const auto RefersToElement =
      declRefExpr(to(parmVarDecl(equalsBoundNode("PredElement"))));
  // The bound is lifted out of the predicate, so it must not name the element.
  const auto Bound = expr(unless(findAll(RefersToElement))).bind("Bound");
  const auto BoundComparison =
      binaryOperator(
          anyOf(allOf(hasAnyOperatorName(">", ">="),
                      hasLHS(ignoringParenImpCasts(RefersToElement)),
                      hasRHS(ignoringParenImpCasts(Bound))),
                allOf(hasAnyOperatorName("<", "<="),
                      hasLHS(ignoringParenImpCasts(Bound)),
                      hasRHS(ignoringParenImpCasts(RefersToElement)))))
          .bind("PredOp");
  // A second parameter or an init-capture would let the bound name a
  // declaration that is not in scope at the call site the fix moves it to.
  const auto BoundPredicate =
      lambdaExpr(
          matchers::hasCallOperator(cxxMethodDecl(
              parameterCountIs(1),
              hasParameter(0, parmVarDecl().bind("PredElement")))),
          unless(hasAnyCapture(capturesVar(varDecl(isInitCapture())))),
          has(compoundStmt(statementCountIs(1),
                           hasAnySubstatement(returnStmt(hasReturnValue(
                               ignoringParenImpCasts(BoundComparison)))))))
          .bind("Pred");

  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::std::find_if"))),
               hasWholeContainerRange(SortedContainer),
               hasArgument(2, ignoringElidableConstructorCall(BoundPredicate)))
          .bind("IneffAlg"),
      this);
}

void InefficientAlgorithmCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("IneffAlg");
  const auto *PredOp = Result.Nodes.getNodeAs<BinaryOperator>("PredOp");

  const Expr *ValueExpr = AlgCall->getArg(2);
  const Expr *ValueExprAsWritten = ValueExpr;
  const Expr *ElementExpr = nullptr;
  if (PredOp) {
    const BinaryOperatorKind Opcode = PredOp->getOpcode();
    const bool ElementOnLeft = Opcode == BO_GT || Opcode == BO_GE;
    ValueExpr = Result.Nodes.getNodeAs<Expr>("Bound");
    ValueExprAsWritten = ElementOnLeft ? PredOp->getRHS() : PredOp->getLHS();
    ElementExpr = ElementOnLeft ? PredOp->getLHS() : PredOp->getRHS();
  }

  const auto *IneffCont =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffCont");
  bool PtrToContainer = false;
  if (!IneffCont) {
    IneffCont =
        Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffContPtr");
    PtrToContainer = true;
  }
  const StringRef IneffContName = IneffCont->getName();
  const bool Unordered = IneffContName.contains("unordered");
  const bool Maplike = IneffContName.contains("map");

  const QualType ValueType = ValueExpr->getType();
  const QualType KeyType =
      IneffCont->getTemplateArgs()[0].getAsType().getCanonicalType();
  bool CompatibleTypes = areTypesCompatible(KeyType, ValueType);

  if (PredOp) {
    if (!hasBuiltinOrder(KeyType))
      return;

    if (ValueExpr->getType().isVolatileQualified() ||
        ValueExpr->HasSideEffects(*Result.Context))
      return;

    // A generic lambda's template parameters are not in scope at the call
    // site, and its dependent body is not reliably spelled as a
    // `binaryOperator`.
    if (Result.Nodes.getNodeAs<LambdaExpr>("Pred")->isGenericLambda())
      return;

    const QualType PredType =
        Result.Nodes.getNodeAs<ParmVarDecl>("PredElement")->getType();
    if (!areTypesCompatible(KeyType, PredType))
      return;

    // Arithmetic conversions can make the predicate compare as unsigned
    // while the container method compares as signed.
    if (KeyType->isSignedIntegerOrEnumerationType() &&
        ElementExpr->getType()->isUnsignedIntegerOrEnumerationType())
      return;

    if (!CompatibleTypes) {
      if (!boundConvertsExactly(ValueExpr, KeyType, *Result.Context))
        return;
      CompatibleTypes = true;
    }
  }

  // Check if the comparison type for the algorithm and the container matches.
  if (AlgCall->getNumArgs() == 4 && !Unordered) {
    const Expr *Arg = AlgCall->getArg(3);
    const QualType AlgCmp =
        Arg->getType().getUnqualifiedType().getCanonicalType();
    const unsigned CmpPosition = IneffContName.contains("map") ? 2 : 1;
    const QualType ContainerCmp = IneffCont->getTemplateArgs()[CmpPosition]
                                      .getAsType()
                                      .getUnqualifiedType()
                                      .getCanonicalType();
    if (AlgCmp != ContainerCmp) {
      diag(Arg->getBeginLoc(),
           "different comparers used in the algorithm and the container");
      return;
    }
  }

  const auto *AlgDecl = AlgCall->getDirectCallee();
  if (!AlgDecl)
    return;

  if (Unordered && AlgDecl->getName().contains("bound"))
    return;

  StringRef MethodName = AlgDecl->getName();
  if (PredOp) {
    const BinaryOperatorKind Opcode = PredOp->getOpcode();
    MethodName =
        Opcode == BO_GT || Opcode == BO_LT ? "upper_bound" : "lower_bound";
  }

  const auto *IneffContExpr = Result.Nodes.getNodeAs<Expr>("IneffContExpr");
  FixItHint Hint;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions LangOpts = getLangOpts();

  CharSourceRange CallRange =
      CharSourceRange::getTokenRange(AlgCall->getSourceRange());

  // FIXME: Create a common utility to extract a file range that the given token
  // sequence is exactly spelled at (without macro argument expansions etc.).
  // We can't use Lexer::makeFileCharRange here, because for
  //
  //   #define F(x) x
  //   x(a b c);
  //
  // it will return "x(a b c)", when given the range "a"-"c". It makes sense for
  // removals, but not for replacements.
  //
  // This code is over-simplified, but works for many real cases.
  if (SM.isMacroArgExpansion(CallRange.getBegin()) &&
      SM.isMacroArgExpansion(CallRange.getEnd())) {
    CallRange.setBegin(SM.getSpellingLoc(CallRange.getBegin()));
    CallRange.setEnd(SM.getSpellingLoc(CallRange.getEnd()));
  }

  if (!CallRange.getBegin().isMacroID() && !Maplike && CompatibleTypes) {
    const StringRef ContainerText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(IneffContExpr->getSourceRange()), SM,
        LangOpts);
    const StringRef ParamText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(ValueExprAsWritten->getSourceRange()),
        SM, LangOpts);
    // There is no source text for an expression that covers only part of a
    // macro expansion. Building the replacement from an empty string would
    // incorrectly drop the container or the value.
    if (!ContainerText.empty() && !ParamText.empty()) {
      const std::string ReplacementText =
          (llvm::Twine(ContainerText) + (PtrToContainer ? "->" : ".") +
           MethodName + "(" + ParamText + ")")
              .str();
      Hint = FixItHint::CreateReplacement(CallRange, ReplacementText);
    }
  }

  diag(AlgCall->getBeginLoc(),
       "this STL algorithm call should be replaced with the container "
       "method '%0'")
      << MethodName << Hint;
}

} // namespace clang::tidy::performance
