//===--- SuspiciousEnumUsageCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousEnumUsageCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

static const char DifferentEnumErrorMessage[] =
    "enum values are from different enum types";

static const char BitmaskErrorMessage[] =
    "enum type seems like a bitmask (contains mostly "
    "power-of-2 literals), but this literal is not a "
    "power-of-2";

static const char BitmaskVarErrorMessage[] =
    "enum type seems like a bitmask (contains mostly "
    "power-of-2 literals) but %plural{1:a literal is|:some literals are}0 not "
    "power-of-2";

static const char BitmaskNoteMessage[] = "used here as a bitmask";

/// Stores a min and a max value which describe an interval.
struct ValueRange {
  llvm::APSInt MinVal;
  llvm::APSInt MaxVal;

  ValueRange(const EnumDecl *EnumDec) {
    const auto MinMaxVal = std::minmax_element(
        EnumDec->enumerator_begin(), EnumDec->enumerator_end(),
        [](const EnumConstantDecl *E1, const EnumConstantDecl *E2) {
          return llvm::APSInt::compareValues(E1->getInitVal(),
                                             E2->getInitVal()) < 0;
        });
    MinVal = MinMaxVal.first->getInitVal();
    MaxVal = MinMaxVal.second->getInitVal();
  }
};

/// Return the number of EnumConstantDecls in an EnumDecl.
static int enumLength(const EnumDecl *EnumDec) {
  return std::distance(EnumDec->enumerator_begin(), EnumDec->enumerator_end());
}

static bool hasDisjointValueRange(const EnumDecl *Enum1,
                                  const EnumDecl *Enum2) {
  ValueRange Range1(Enum1), Range2(Enum2);
  return llvm::APSInt::compareValues(Range1.MaxVal, Range2.MinVal) < 0 ||
         llvm::APSInt::compareValues(Range2.MaxVal, Range1.MinVal) < 0;
}

static bool isNonPowerOf2NorNullLiteral(const EnumConstantDecl *EnumConst) {
  llvm::APSInt Val = EnumConst->getInitVal();
  if (Val.isPowerOf2() || !Val.getBoolValue())
    return false;
  const Expr *InitExpr = EnumConst->getInitExpr();
  if (!InitExpr)
    return true;
  return isa<IntegerLiteral>(InitExpr->IgnoreImpCasts());
}

static bool isMaxValAllBitSetLiteral(const EnumDecl *EnumDec) {
  auto EnumConst = std::max_element(
      EnumDec->enumerator_begin(), EnumDec->enumerator_end(),
      [](const EnumConstantDecl *E1, const EnumConstantDecl *E2) {
        return E1->getInitVal() < E2->getInitVal();
      });

  if (const Expr *InitExpr = EnumConst->getInitExpr()) {
    return EnumConst->getInitVal().countTrailingOnes() ==
               EnumConst->getInitVal().getActiveBits() &&
           isa<IntegerLiteral>(InitExpr->IgnoreImpCasts());
  }
  return false;
}

static int countNonPowOfTwoLiteralNum(const EnumDecl *EnumDec) {
  return std::count_if(
      EnumDec->enumerator_begin(), EnumDec->enumerator_end(),
      [](const EnumConstantDecl *E) { return isNonPowerOf2NorNullLiteral(E); });
}

/// Check if there is one or two enumerators that are not a power of 2 and are
/// initialized by a literal in the enum type, and that the enumeration contains
/// enough elements to reasonably act as a bitmask. Exclude the case where the
/// last enumerator is the sum of the lesser values (and initialized by a
/// literal) or when it could contain consecutive values.
static bool isPossiblyBitMask(const EnumDecl *EnumDec) {
  ValueRange VR(EnumDec);
  int EnumLen = enumLength(EnumDec);
  int NonPowOfTwoCounter = countNonPowOfTwoLiteralNum(EnumDec);
  return NonPowOfTwoCounter >= 1 && NonPowOfTwoCounter <= 2 &&
         NonPowOfTwoCounter < EnumLen / 2 &&
         (VR.MaxVal - VR.MinVal != EnumLen - 1) &&
         !(NonPowOfTwoCounter == 1 && isMaxValAllBitSetLiteral(EnumDec));
}

SuspiciousEnumUsageCheck::SuspiciousEnumUsageCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)) {}

void SuspiciousEnumUsageCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void SuspiciousEnumUsageCheck::registerMatchers(MatchFinder *Finder) {
  const auto EnumExpr = [](StringRef RefName, StringRef DeclName) {
    return expr(hasType(enumDecl().bind(DeclName))).bind(RefName);
  };

  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("|"), hasLHS(hasType(enumDecl().bind("enumDecl"))),
          hasRHS(hasType(enumDecl(unless(equalsBoundNode("enumDecl")))
                             .bind("otherEnumDecl"))))
          .bind("diffEnumOp"),
      this);

  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("+", "|"),
                     hasLHS(EnumExpr("lhsExpr", "enumDecl")),
                     hasRHS(expr(hasType(enumDecl(equalsBoundNode("enumDecl"))))
                                .bind("rhsExpr"))),
      this);

  Finder->addMatcher(
      binaryOperator(
          hasAnyOperatorName("+", "|"),
          hasOperands(expr(hasType(isInteger()), unless(hasType(enumDecl()))),
                      EnumExpr("enumExpr", "enumDecl"))),
      this);

  Finder->addMatcher(binaryOperator(hasAnyOperatorName("|=", "+="),
                                    hasRHS(EnumExpr("enumExpr", "enumDecl"))),
                     this);
}

void SuspiciousEnumUsageCheck::checkSuspiciousBitmaskUsage(
    const Expr *NodeExpr, const EnumDecl *EnumDec) {
  const auto *EnumExpr = dyn_cast<DeclRefExpr>(NodeExpr);
  const auto *EnumConst =
      EnumExpr ? dyn_cast<EnumConstantDecl>(EnumExpr->getDecl()) : nullptr;

  // Report the parameter if necessary.
  if (!EnumConst) {
    diag(EnumDec->getInnerLocStart(), BitmaskVarErrorMessage)
        << countNonPowOfTwoLiteralNum(EnumDec);
    diag(EnumExpr->getExprLoc(), BitmaskNoteMessage, DiagnosticIDs::Note);
  } else if (isNonPowerOf2NorNullLiteral(EnumConst)) {
    diag(EnumConst->getSourceRange().getBegin(), BitmaskErrorMessage);
    diag(EnumExpr->getExprLoc(), BitmaskNoteMessage, DiagnosticIDs::Note);
  }
}

void SuspiciousEnumUsageCheck::check(const MatchFinder::MatchResult &Result) {
  // Case 1: The two enum values come from different types.
  if (const auto *DiffEnumOp =
          Result.Nodes.getNodeAs<BinaryOperator>("diffEnumOp")) {
    const auto *EnumDec = Result.Nodes.getNodeAs<EnumDecl>("enumDecl");
    const auto *OtherEnumDec =
        Result.Nodes.getNodeAs<EnumDecl>("otherEnumDecl");
    // Skip when one of the parameters is an empty enum. The
    // hasDisjointValueRange function could not decide the values properly in
    // case of an empty enum.
    if (EnumDec->enumerator_begin() == EnumDec->enumerator_end() ||
        OtherEnumDec->enumerator_begin() == OtherEnumDec->enumerator_end())
      return;

    if (!hasDisjointValueRange(EnumDec, OtherEnumDec))
      diag(DiffEnumOp->getOperatorLoc(), DifferentEnumErrorMessage);
    return;
  }

  // Case 2 and 3 only checked in strict mode. The checker tries to detect
  // suspicious bitmasks which contains values initialized by non power-of-2
  // literals.
  if (!StrictMode)
    return;
  const auto *EnumDec = Result.Nodes.getNodeAs<EnumDecl>("enumDecl");
  if (!isPossiblyBitMask(EnumDec))
    return;

  // Case 2:
  //   a. Investigating the right hand side of `+=` or `|=` operator.
  //   b. When the operator is `|` or `+` but only one of them is an EnumExpr
  if (const auto *EnumExpr = Result.Nodes.getNodeAs<Expr>("enumExpr")) {
    checkSuspiciousBitmaskUsage(EnumExpr, EnumDec);
    return;
  }

  // Case 3:
  // '|' or '+' operator where both argument comes from the same enum type
  const auto *LhsExpr = Result.Nodes.getNodeAs<Expr>("lhsExpr");
  checkSuspiciousBitmaskUsage(LhsExpr, EnumDec);

  const auto *RhsExpr = Result.Nodes.getNodeAs<Expr>("rhsExpr");
  checkSuspiciousBitmaskUsage(RhsExpr, EnumDec);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
