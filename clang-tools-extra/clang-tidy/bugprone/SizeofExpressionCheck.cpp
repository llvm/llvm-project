//===--- SizeofExpressionCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SizeofExpressionCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P(IntegerLiteral, isBiggerThan, unsigned, N) {
  return Node.getValue().ugt(N);
}

AST_MATCHER_P2(Expr, hasSizeOfDescendant, int, Depth,
               ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  if (Depth < 0)
    return false;

  const Expr *E = Node.IgnoreParenImpCasts();
  if (InnerMatcher.matches(*E, Finder, Builder))
    return true;

  if (const auto *CE = dyn_cast<CastExpr>(E)) {
    const auto M = hasSizeOfDescendant(Depth - 1, InnerMatcher);
    return M.matches(*CE->getSubExpr(), Finder, Builder);
  }
  if (const auto *UE = dyn_cast<UnaryOperator>(E)) {
    const auto M = hasSizeOfDescendant(Depth - 1, InnerMatcher);
    return M.matches(*UE->getSubExpr(), Finder, Builder);
  }
  if (const auto *BE = dyn_cast<BinaryOperator>(E)) {
    const auto LHS = hasSizeOfDescendant(Depth - 1, InnerMatcher);
    const auto RHS = hasSizeOfDescendant(Depth - 1, InnerMatcher);
    return LHS.matches(*BE->getLHS(), Finder, Builder) ||
           RHS.matches(*BE->getRHS(), Finder, Builder);
  }

  return false;
}

CharUnits getSizeOfType(const ASTContext &Ctx, const Type *Ty) {
  if (!Ty || Ty->isIncompleteType() || Ty->isDependentType() ||
      isa<DependentSizedArrayType>(Ty) || !Ty->isConstantSizeType())
    return CharUnits::Zero();
  return Ctx.getTypeSizeInChars(Ty);
}

} // namespace

SizeofExpressionCheck::SizeofExpressionCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnSizeOfConstant(Options.get("WarnOnSizeOfConstant", true)),
      WarnOnSizeOfIntegerExpression(
          Options.get("WarnOnSizeOfIntegerExpression", false)),
      WarnOnSizeOfThis(Options.get("WarnOnSizeOfThis", true)),
      WarnOnSizeOfCompareToConstant(
          Options.get("WarnOnSizeOfCompareToConstant", true)),
      WarnOnSizeOfPointerToAggregate(
          Options.get("WarnOnSizeOfPointerToAggregate", true)),
      WarnOnSizeOfPointer(Options.get("WarnOnSizeOfPointer", false)) {}

void SizeofExpressionCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnSizeOfConstant", WarnOnSizeOfConstant);
  Options.store(Opts, "WarnOnSizeOfIntegerExpression",
                WarnOnSizeOfIntegerExpression);
  Options.store(Opts, "WarnOnSizeOfThis", WarnOnSizeOfThis);
  Options.store(Opts, "WarnOnSizeOfCompareToConstant",
                WarnOnSizeOfCompareToConstant);
  Options.store(Opts, "WarnOnSizeOfPointerToAggregate",
                WarnOnSizeOfPointerToAggregate);
  Options.store(Opts, "WarnOnSizeOfPointer", WarnOnSizeOfPointer);
}

void SizeofExpressionCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME:
  // Some of the checks should not match in template code to avoid false
  // positives if sizeof is applied on template argument.

  const auto IntegerExpr = ignoringParenImpCasts(integerLiteral());
  const auto ConstantExpr = ignoringParenImpCasts(
      anyOf(integerLiteral(), unaryOperator(hasUnaryOperand(IntegerExpr)),
            binaryOperator(hasLHS(IntegerExpr), hasRHS(IntegerExpr))));
  const auto IntegerCallExpr = ignoringParenImpCasts(callExpr(
      anyOf(hasType(isInteger()), hasType(hasCanonicalType(enumType()))),
      unless(isInTemplateInstantiation())));
  const auto SizeOfExpr = sizeOfExpr(hasArgumentOfType(
      hasUnqualifiedDesugaredType(type().bind("sizeof-arg-type"))));
  const auto SizeOfZero =
      sizeOfExpr(has(ignoringParenImpCasts(integerLiteral(equals(0)))));

  // Detect expression like: sizeof(ARRAYLEN);
  // Note: The expression 'sizeof(sizeof(0))' is a portable trick used to know
  //       the sizeof size_t.
  if (WarnOnSizeOfConstant) {
    Finder->addMatcher(
        expr(sizeOfExpr(has(ignoringParenImpCasts(ConstantExpr))),
             unless(SizeOfZero))
            .bind("sizeof-constant"),
        this);
  }

  // Detect sizeof(f())
  if (WarnOnSizeOfIntegerExpression) {
    Finder->addMatcher(sizeOfExpr(ignoringParenImpCasts(has(IntegerCallExpr)))
                           .bind("sizeof-integer-call"),
                       this);
  }

  // Detect expression like: sizeof(this);
  if (WarnOnSizeOfThis) {
    Finder->addMatcher(sizeOfExpr(has(ignoringParenImpCasts(cxxThisExpr())))
                           .bind("sizeof-this"),
                       this);
  }

  // Detect sizeof(kPtr) where kPtr is 'const char* kPtr = "abc"';
  const auto CharPtrType = pointerType(pointee(isAnyCharacter()));
  const auto ConstStrLiteralDecl =
      varDecl(isDefinition(), hasType(hasCanonicalType(CharPtrType)),
              hasInitializer(ignoringParenImpCasts(stringLiteral())));
  const auto VarWithConstStrLiteralDecl = expr(
      hasType(hasCanonicalType(CharPtrType)),
      ignoringParenImpCasts(declRefExpr(hasDeclaration(ConstStrLiteralDecl))));
  Finder->addMatcher(
      sizeOfExpr(has(ignoringParenImpCasts(VarWithConstStrLiteralDecl)))
          .bind("sizeof-charp"),
      this);

  // Detect sizeof(ptr) where ptr is a pointer (CWE-467).
  //
  // In WarnOnSizeOfPointerToAggregate mode only report cases when ptr points
  // to an aggregate type or ptr is an expression that (implicitly or
  // explicitly) casts an array to a pointer type. (These are more suspicious
  // than other sizeof(ptr) expressions because they can appear as distorted
  // forms of the common sizeof(aggregate) expressions.)
  //
  // To avoid false positives, the check doesn't report expressions like
  // 'sizeof(pp[0])' and 'sizeof(*pp)' where `pp` is a pointer-to-pointer or
  // array of pointers. (This filters out both `sizeof(arr) / sizeof(arr[0])`
  // expressions and other cases like `p = realloc(p, newsize * sizeof(*p));`.)
  //
  // Moreover this generic message is suppressed in cases that are also matched
  // by the more concrete matchers 'sizeof-this' and 'sizeof-charp'.
  if (WarnOnSizeOfPointerToAggregate || WarnOnSizeOfPointer) {
    const auto ArrayExpr =
        ignoringParenImpCasts(hasType(hasCanonicalType(arrayType())));
    const auto ArrayCastExpr = expr(anyOf(
        unaryOperator(hasUnaryOperand(ArrayExpr), unless(hasOperatorName("*"))),
        binaryOperator(hasEitherOperand(ArrayExpr)),
        castExpr(hasSourceExpression(ArrayExpr))));
    const auto PointerToArrayExpr =
        hasType(hasCanonicalType(pointerType(pointee(arrayType()))));

    const auto PointerToStructType =
        hasUnqualifiedDesugaredType(pointerType(pointee(recordType())));
    const auto PointerToStructTypeWithBinding =
        type(PointerToStructType).bind("struct-type");
    const auto PointerToStructExpr =
        expr(hasType(hasCanonicalType(PointerToStructType)));

    const auto PointerToDetectedExpr =
        WarnOnSizeOfPointer
            ? expr(hasType(hasUnqualifiedDesugaredType(pointerType())))
            : expr(anyOf(ArrayCastExpr, PointerToArrayExpr,
                         PointerToStructExpr));

    const auto ZeroLiteral = ignoringParenImpCasts(integerLiteral(equals(0)));
    const auto SubscriptExprWithZeroIndex =
        arraySubscriptExpr(hasIndex(ZeroLiteral));
    const auto DerefExpr =
        ignoringParenImpCasts(unaryOperator(hasOperatorName("*")));

    Finder->addMatcher(
        expr(sizeOfExpr(anyOf(has(ignoringParenImpCasts(
                                  expr(PointerToDetectedExpr, unless(DerefExpr),
                                       unless(SubscriptExprWithZeroIndex),
                                       unless(VarWithConstStrLiteralDecl),
                                       unless(cxxThisExpr())))),
                              has(PointerToStructTypeWithBinding))))
            .bind("sizeof-pointer"),
        this);
  }

  // Detect expression like: sizeof(expr) <= k for a suspicious constant 'k'.
  if (WarnOnSizeOfCompareToConstant) {
    Finder->addMatcher(
        binaryOperator(matchers::isRelationalOperator(),
                       hasOperands(ignoringParenImpCasts(SizeOfExpr),
                                   ignoringParenImpCasts(integerLiteral(anyOf(
                                       equals(0), isBiggerThan(0x80000))))))
            .bind("sizeof-compare-constant"),
        this);
  }

  // Detect expression like: sizeof(expr, expr); most likely an error.
  Finder->addMatcher(
      sizeOfExpr(
          has(ignoringParenImpCasts(
              binaryOperator(hasOperatorName(",")).bind("sizeof-comma-binop"))))
          .bind("sizeof-comma-expr"),
      this);

  // Detect sizeof(...) /sizeof(...));
  // FIXME:
  // Re-evaluate what cases to handle by the checker.
  // Probably any sizeof(A)/sizeof(B) should be error if
  // 'A' is not an array (type) and 'B' the (type of the) first element of it.
  // Except if 'A' and 'B' are non-pointers, then use the existing size division
  // rule.
  const auto ElemType =
      arrayType(hasElementType(recordType().bind("elem-type")));
  const auto ElemPtrType = pointerType(pointee(type().bind("elem-ptr-type")));

  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("/"),
          hasLHS(ignoringParenImpCasts(sizeOfExpr(hasArgumentOfType(
              hasCanonicalType(type(anyOf(ElemType, ElemPtrType, type()))
                                   .bind("num-type")))))),
          hasRHS(ignoringParenImpCasts(sizeOfExpr(
              hasArgumentOfType(hasCanonicalType(type().bind("denom-type")))))))
          .bind("sizeof-divide-expr"),
      this);

  // Detect expression like: sizeof(...) * sizeof(...)); most likely an error.
  Finder->addMatcher(binaryOperator(hasOperatorName("*"),
                                    hasLHS(ignoringParenImpCasts(SizeOfExpr)),
                                    hasRHS(ignoringParenImpCasts(SizeOfExpr)))
                         .bind("sizeof-multiply-sizeof"),
                     this);

  Finder->addMatcher(
      binaryOperator(hasOperatorName("*"),
                     hasOperands(ignoringParenImpCasts(SizeOfExpr),
                                 ignoringParenImpCasts(binaryOperator(
                                     hasOperatorName("*"),
                                     hasEitherOperand(
                                         ignoringParenImpCasts(SizeOfExpr))))))
          .bind("sizeof-multiply-sizeof"),
      this);

  // Detect strange double-sizeof expression like: sizeof(sizeof(...));
  // Note: The expression 'sizeof(sizeof(0))' is accepted.
  Finder->addMatcher(sizeOfExpr(has(ignoringParenImpCasts(hasSizeOfDescendant(
                                    8, allOf(SizeOfExpr, unless(SizeOfZero))))))
                         .bind("sizeof-sizeof-expr"),
                     this);

  // Detect sizeof in pointer arithmetic like: N * sizeof(S) == P1 - P2 or
  // (P1 - P2) / sizeof(S) where P1 and P2 are pointers to type S.
  const auto PtrDiffExpr = binaryOperator(
      hasOperatorName("-"),
      hasLHS(hasType(hasUnqualifiedDesugaredType(pointerType(pointee(
          hasUnqualifiedDesugaredType(type().bind("left-ptr-type"))))))),
      hasRHS(hasType(hasUnqualifiedDesugaredType(pointerType(pointee(
          hasUnqualifiedDesugaredType(type().bind("right-ptr-type"))))))));

  Finder->addMatcher(
      binaryOperator(
          hasAnyOperatorName("==", "!=", "<", "<=", ">", ">=", "+", "-"),
          hasOperands(anyOf(ignoringParenImpCasts(
                                SizeOfExpr.bind("sizeof-ptr-mul-expr")),
                            ignoringParenImpCasts(binaryOperator(
                                hasOperatorName("*"),
                                hasEitherOperand(ignoringParenImpCasts(
                                    SizeOfExpr.bind("sizeof-ptr-mul-expr")))))),
                      ignoringParenImpCasts(PtrDiffExpr)))
          .bind("sizeof-in-ptr-arithmetic-mul"),
      this);

  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("/"), hasLHS(ignoringParenImpCasts(PtrDiffExpr)),
          hasRHS(ignoringParenImpCasts(SizeOfExpr.bind("sizeof-ptr-div-expr"))))
          .bind("sizeof-in-ptr-arithmetic-div"),
      this);
}

void SizeofExpressionCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;

  if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-constant")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof(K)'; did you mean 'K'?")
        << E->getSourceRange();
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-integer-call")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof()' on an expression "
                           "that results in an integer")
        << E->getSourceRange();
  } else if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-this")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(this)'; did you mean 'sizeof(*this)'")
        << E->getSourceRange();
  } else if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-charp")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(char*)'; do you mean 'strlen'?")
        << E->getSourceRange();
  } else if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-pointer")) {
    if (Result.Nodes.getNodeAs<Type>("struct-type")) {
      diag(E->getBeginLoc(),
           "suspicious usage of 'sizeof(A*)' on pointer-to-aggregate type; did "
           "you mean 'sizeof(A)'?")
          << E->getSourceRange();
    } else {
      diag(E->getBeginLoc(), "suspicious usage of 'sizeof()' on an expression "
                             "that results in a pointer")
          << E->getSourceRange();
    }
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "sizeof-compare-constant")) {
    diag(E->getOperatorLoc(),
         "suspicious comparison of 'sizeof(expr)' to a constant")
        << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-comma-expr")) {
    const auto *BO =
        Result.Nodes.getNodeAs<BinaryOperator>("sizeof-comma-binop");
    assert(BO);
    diag(BO->getOperatorLoc(), "suspicious usage of 'sizeof(..., ...)'")
        << E->getSourceRange();
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<BinaryOperator>("sizeof-divide-expr")) {
    const auto *NumTy = Result.Nodes.getNodeAs<Type>("num-type");
    const auto *DenomTy = Result.Nodes.getNodeAs<Type>("denom-type");
    const auto *ElementTy = Result.Nodes.getNodeAs<Type>("elem-type");
    const auto *PointedTy = Result.Nodes.getNodeAs<Type>("elem-ptr-type");

    CharUnits NumeratorSize = getSizeOfType(Ctx, NumTy);
    CharUnits DenominatorSize = getSizeOfType(Ctx, DenomTy);
    CharUnits ElementSize = getSizeOfType(Ctx, ElementTy);

    if (DenominatorSize > CharUnits::Zero() &&
        !NumeratorSize.isMultipleOf(DenominatorSize)) {
      diag(E->getOperatorLoc(), "suspicious usage of 'sizeof(...)/sizeof(...)';"
                                " numerator is not a multiple of denominator")
          << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
    } else if (ElementSize > CharUnits::Zero() &&
               DenominatorSize > CharUnits::Zero() &&
               ElementSize != DenominatorSize) {
      diag(E->getOperatorLoc(), "suspicious usage of 'sizeof(...)/sizeof(...)';"
                                " numerator is not a multiple of denominator")
          << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
    } else if (NumTy && DenomTy && NumTy == DenomTy) {
      // FIXME: This message is wrong, it should not refer to sizeof "pointer"
      // usage (and by the way, it would be to clarify all the messages).
      diag(E->getOperatorLoc(),
           "suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'")
          << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
    } else if (!WarnOnSizeOfPointer) {
      // When 'WarnOnSizeOfPointer' is enabled, these messages become redundant:
      if (PointedTy && DenomTy && PointedTy == DenomTy) {
        diag(E->getOperatorLoc(),
             "suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'")
            << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
      } else if (NumTy && DenomTy && NumTy->isPointerType() &&
                 DenomTy->isPointerType()) {
        diag(E->getOperatorLoc(),
             "suspicious usage of sizeof pointer 'sizeof(P*)/sizeof(Q*)'")
            << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
      }
    }
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-sizeof-expr")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof(sizeof(...))'")
        << E->getSourceRange();
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "sizeof-multiply-sizeof")) {
    diag(E->getOperatorLoc(), "suspicious 'sizeof' by 'sizeof' multiplication")
        << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "sizeof-in-ptr-arithmetic-mul")) {
    const auto *LPtrTy = Result.Nodes.getNodeAs<Type>("left-ptr-type");
    const auto *RPtrTy = Result.Nodes.getNodeAs<Type>("right-ptr-type");
    const auto *SizeofArgTy = Result.Nodes.getNodeAs<Type>("sizeof-arg-type");
    const auto *SizeOfExpr =
        Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof-ptr-mul-expr");

    if ((LPtrTy == RPtrTy) && (LPtrTy == SizeofArgTy)) {
      diag(SizeOfExpr->getBeginLoc(), "suspicious usage of 'sizeof(...)' in "
                                      "pointer arithmetic")
          << SizeOfExpr->getSourceRange() << E->getOperatorLoc()
          << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
    }
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "sizeof-in-ptr-arithmetic-div")) {
    const auto *LPtrTy = Result.Nodes.getNodeAs<Type>("left-ptr-type");
    const auto *RPtrTy = Result.Nodes.getNodeAs<Type>("right-ptr-type");
    const auto *SizeofArgTy = Result.Nodes.getNodeAs<Type>("sizeof-arg-type");
    const auto *SizeOfExpr =
        Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof-ptr-div-expr");

    if ((LPtrTy == RPtrTy) && (LPtrTy == SizeofArgTy)) {
      diag(SizeOfExpr->getBeginLoc(), "suspicious usage of 'sizeof(...)' in "
                                      "pointer arithmetic")
          << SizeOfExpr->getSourceRange() << E->getOperatorLoc()
          << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
    }
  }
}

} // namespace clang::tidy::bugprone
