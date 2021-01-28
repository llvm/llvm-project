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

namespace clang {
namespace tidy {
namespace bugprone {

namespace {

AST_MATCHER_P(IntegerLiteral, isBiggerThan, unsigned, N) {
  return Node.getValue().getZExtValue() > N;
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
          Options.get("WarnOnSizeOfCompareToConstant", true)) {}

void SizeofExpressionCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnSizeOfConstant", WarnOnSizeOfConstant);
  Options.store(Opts, "WarnOnSizeOfIntegerExpression",
                WarnOnSizeOfIntegerExpression);
  Options.store(Opts, "WarnOnSizeOfThis", WarnOnSizeOfThis);
  Options.store(Opts, "WarnOnSizeOfCompareToConstant",
                WarnOnSizeOfCompareToConstant);
}

void SizeofExpressionCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME:
  // Some of the checks should not match in template code to avoid false
  // positives if sizeof is applied on template argument.

  const auto IntegerExpr = ignoringParenImpCasts(integerLiteral());
  const auto ConstantExpr = expr(ignoringParenImpCasts(
      anyOf(integerLiteral(), unaryOperator(hasUnaryOperand(IntegerExpr)),
            binaryOperator(hasLHS(IntegerExpr), hasRHS(IntegerExpr)))));
  const auto IntegerCallExpr = expr(ignoringParenImpCasts(
      callExpr(anyOf(hasType(isInteger()), hasType(enumType())),
               unless(isInTemplateInstantiation()))));
  const auto SizeOfExpr = expr(anyOf(
      sizeOfExpr(
          has(hasUnqualifiedDesugaredType(type().bind("sizeof-arg-type")))),
      sizeOfExpr(has(expr(hasType(
          hasUnqualifiedDesugaredType(type().bind("sizeof-arg-type"))))))));
  const auto SizeOfZero = expr(
      sizeOfExpr(has(ignoringParenImpCasts(expr(integerLiteral(equals(0)))))));

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
    Finder->addMatcher(
        expr(sizeOfExpr(ignoringParenImpCasts(has(IntegerCallExpr))))
            .bind("sizeof-integer-call"),
        this);
  }

  // Detect expression like: sizeof(this);
  if (WarnOnSizeOfThis) {
    Finder->addMatcher(
        expr(sizeOfExpr(has(ignoringParenImpCasts(expr(cxxThisExpr())))))
            .bind("sizeof-this"),
        this);
  }

  // Detect sizeof(kPtr) where kPtr is 'const char* kPtr = "abc"';
  const auto CharPtrType = pointerType(pointee(isAnyCharacter()));
  const auto ConstStrLiteralDecl =
      varDecl(isDefinition(), hasType(qualType(hasCanonicalType(CharPtrType))),
              hasInitializer(ignoringParenImpCasts(stringLiteral())));
  Finder->addMatcher(expr(sizeOfExpr(has(ignoringParenImpCasts(expr(
                              hasType(qualType(hasCanonicalType(CharPtrType))),
                              ignoringParenImpCasts(declRefExpr(
                                  hasDeclaration(ConstStrLiteralDecl))))))))
                         .bind("sizeof-charp"),
                     this);

  // Detect sizeof(ptr) where ptr points to an aggregate (i.e. sizeof(&S)).
  // Do not find it if RHS of a 'sizeof(arr) / sizeof(arr[0])' expression.
  const auto ArrayExpr = expr(ignoringParenImpCasts(
      expr(hasType(qualType(hasCanonicalType(arrayType()))))));
  const auto ArrayCastExpr = expr(anyOf(
      unaryOperator(hasUnaryOperand(ArrayExpr), unless(hasOperatorName("*"))),
      binaryOperator(hasEitherOperand(ArrayExpr)),
      castExpr(hasSourceExpression(ArrayExpr))));
  const auto PointerToArrayExpr = expr(ignoringParenImpCasts(expr(
      hasType(qualType(hasCanonicalType(pointerType(pointee(arrayType()))))))));

  const auto StructAddrOfExpr =
      unaryOperator(hasOperatorName("&"),
                    hasUnaryOperand(ignoringParenImpCasts(expr(
                        hasType(qualType(hasCanonicalType(recordType())))))));
  const auto PointerToStructType = type(hasUnqualifiedDesugaredType(
      pointerType(pointee(recordType()))));
  const auto PointerToStructExpr = expr(ignoringParenImpCasts(expr(
      hasType(qualType(hasCanonicalType(PointerToStructType))),
      unless(cxxThisExpr()))));

  const auto ArrayOfPointersExpr = expr(ignoringParenImpCasts(expr(hasType(
      qualType(hasCanonicalType(arrayType(hasElementType(pointerType()))
                                    .bind("type-of-array-of-pointers")))))));
  const auto ArrayOfSamePointersExpr =
      expr(ignoringParenImpCasts(expr(hasType(qualType(hasCanonicalType(
          arrayType(equalsBoundNode("type-of-array-of-pointers"))))))));
  const auto ZeroLiteral =
      expr(ignoringParenImpCasts(integerLiteral(equals(0))));
  const auto ArrayOfSamePointersZeroSubscriptExpr =
      expr(ignoringParenImpCasts(arraySubscriptExpr(
          hasBase(ArrayOfSamePointersExpr), hasIndex(ZeroLiteral))));
  const auto ArrayLengthExprDenom =
      expr(hasParent(expr(ignoringParenImpCasts(
               binaryOperator(hasOperatorName("/"),
                              hasLHS(expr(ignoringParenImpCasts(expr(
                                  sizeOfExpr(has(ArrayOfPointersExpr)))))))))),
           sizeOfExpr(has(ArrayOfSamePointersZeroSubscriptExpr)));

  Finder->addMatcher(expr(anyOf(sizeOfExpr(has(expr(ignoringParenImpCasts(anyOf(
                                    ArrayCastExpr, PointerToArrayExpr,
                                    StructAddrOfExpr, PointerToStructExpr))))),
                                sizeOfExpr(has(PointerToStructType))),
                          unless(ArrayLengthExprDenom))
                         .bind("sizeof-pointer-to-aggregate"),
                     this);

  // Detect expression like: sizeof(epxr) <= k for a suspicious constant 'k'.
  if (WarnOnSizeOfCompareToConstant) {
    Finder->addMatcher(
        binaryOperator(matchers::isRelationalOperator(),
                       hasOperands(ignoringParenImpCasts(SizeOfExpr),
                                   ignoringParenImpCasts(anyOf(
                                       integerLiteral(equals(0)),
                                       integerLiteral(isBiggerThan(0x80000))))))
            .bind("sizeof-compare-constant"),
        this);
  }

  // Detect expression like: sizeof(expr, expr); most likely an error.
  Finder->addMatcher(expr(sizeOfExpr(has(expr(ignoringParenImpCasts(
                              binaryOperator(hasOperatorName(",")))))))
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
  const auto NumType = qualType(hasCanonicalType(
      type(anyOf(ElemType, ElemPtrType, type())).bind("num-type")));
  const auto DenomType = qualType(hasCanonicalType(type().bind("denom-type")));

  Finder->addMatcher(
      binaryOperator(hasOperatorName("/"),
                     hasLHS(expr(ignoringParenImpCasts(
                         anyOf(sizeOfExpr(has(NumType)),
                               sizeOfExpr(has(expr(hasType(NumType)))))))),
                     hasRHS(expr(ignoringParenImpCasts(
                         anyOf(sizeOfExpr(has(DenomType)),
                               sizeOfExpr(has(expr(hasType(DenomType)))))))))
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
  Finder->addMatcher(
      expr(sizeOfExpr(has(ignoringParenImpCasts(expr(
               hasSizeOfDescendant(8, expr(SizeOfExpr, unless(SizeOfZero))))))))
          .bind("sizeof-sizeof-expr"),
      this);

  // Detect sizeof in pointer arithmetic like: N * sizeof(S) == P1 - P2 or
  // (P1 - P2) / sizeof(S) where P1 and P2 are pointers to type S.
  const auto PtrDiffExpr = binaryOperator(
      hasOperatorName("-"),
      hasLHS(expr(hasType(hasUnqualifiedDesugaredType(pointerType(pointee(
          hasUnqualifiedDesugaredType(type().bind("left-ptr-type")))))))),
      hasRHS(expr(hasType(hasUnqualifiedDesugaredType(pointerType(pointee(
          hasUnqualifiedDesugaredType(type().bind("right-ptr-type")))))))));

  Finder->addMatcher(
      binaryOperator(
          hasAnyOperatorName("==", "!=", "<", "<=", ">", ">=", "+", "-"),
          hasOperands(expr(anyOf(ignoringParenImpCasts(SizeOfExpr),
                                 ignoringParenImpCasts(binaryOperator(
                                     hasOperatorName("*"),
                                     hasEitherOperand(
                                         ignoringParenImpCasts(SizeOfExpr)))))),
                      ignoringParenImpCasts(PtrDiffExpr)))
          .bind("sizeof-in-ptr-arithmetic-mul"),
      this);

  Finder->addMatcher(binaryOperator(hasOperatorName("/"),
                                    hasLHS(ignoringParenImpCasts(PtrDiffExpr)),
                                    hasRHS(ignoringParenImpCasts(SizeOfExpr)))
                         .bind("sizeof-in-ptr-arithmetic-div"),
                     this);
}

void SizeofExpressionCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;

  if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-constant")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(K)'; did you mean 'K'?");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-integer-call")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof()' on an expression "
                           "that results in an integer");
  } else if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-this")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(this)'; did you mean 'sizeof(*this)'");
  } else if (const auto *E = Result.Nodes.getNodeAs<Expr>("sizeof-charp")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(char*)'; do you mean 'strlen'?");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-pointer-to-aggregate")) {
    diag(E->getBeginLoc(),
         "suspicious usage of 'sizeof(A*)'; pointer to aggregate");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-compare-constant")) {
    diag(E->getBeginLoc(),
         "suspicious comparison of 'sizeof(expr)' to a constant");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-comma-expr")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof(..., ...)'");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-divide-expr")) {
    const auto *NumTy = Result.Nodes.getNodeAs<Type>("num-type");
    const auto *DenomTy = Result.Nodes.getNodeAs<Type>("denom-type");
    const auto *ElementTy = Result.Nodes.getNodeAs<Type>("elem-type");
    const auto *PointedTy = Result.Nodes.getNodeAs<Type>("elem-ptr-type");

    CharUnits NumeratorSize = getSizeOfType(Ctx, NumTy);
    CharUnits DenominatorSize = getSizeOfType(Ctx, DenomTy);
    CharUnits ElementSize = getSizeOfType(Ctx, ElementTy);

    if (DenominatorSize > CharUnits::Zero() &&
        !NumeratorSize.isMultipleOf(DenominatorSize)) {
      diag(E->getBeginLoc(), "suspicious usage of 'sizeof(...)/sizeof(...)';"
                             " numerator is not a multiple of denominator");
    } else if (ElementSize > CharUnits::Zero() &&
               DenominatorSize > CharUnits::Zero() &&
               ElementSize != DenominatorSize) {
      diag(E->getBeginLoc(), "suspicious usage of 'sizeof(...)/sizeof(...)';"
                             " numerator is not a multiple of denominator");
    } else if (NumTy && DenomTy && NumTy == DenomTy) {
      diag(E->getBeginLoc(),
           "suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'");
    } else if (PointedTy && DenomTy && PointedTy == DenomTy) {
      diag(E->getBeginLoc(),
           "suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'");
    } else if (NumTy && DenomTy && NumTy->isPointerType() &&
               DenomTy->isPointerType()) {
      diag(E->getBeginLoc(),
           "suspicious usage of sizeof pointer 'sizeof(P*)/sizeof(Q*)'");
    }
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-sizeof-expr")) {
    diag(E->getBeginLoc(), "suspicious usage of 'sizeof(sizeof(...))'");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-multiply-sizeof")) {
    diag(E->getBeginLoc(), "suspicious 'sizeof' by 'sizeof' multiplication");
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-in-ptr-arithmetic-mul")) {
    const auto *LPtrTy = Result.Nodes.getNodeAs<Type>("left-ptr-type");
    const auto *RPtrTy = Result.Nodes.getNodeAs<Type>("right-ptr-type");
    const auto *SizeofArgTy = Result.Nodes.getNodeAs<Type>("sizeof-arg-type");

    if ((LPtrTy == RPtrTy) && (LPtrTy == SizeofArgTy)) {
      diag(E->getBeginLoc(), "suspicious usage of 'sizeof(...)' in "
                              "pointer arithmetic");
    }
  } else if (const auto *E =
                 Result.Nodes.getNodeAs<Expr>("sizeof-in-ptr-arithmetic-div")) {
    const auto *LPtrTy = Result.Nodes.getNodeAs<Type>("left-ptr-type");
    const auto *RPtrTy = Result.Nodes.getNodeAs<Type>("right-ptr-type");
    const auto *SizeofArgTy = Result.Nodes.getNodeAs<Type>("sizeof-arg-type");

    if ((LPtrTy == RPtrTy) && (LPtrTy == SizeofArgTy)) {
      diag(E->getBeginLoc(), "suspicious usage of 'sizeof(...)' in "
                              "pointer arithmetic");
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
