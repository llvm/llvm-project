//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdBitCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

UseStdBitCheck::UseStdBitCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      HonorIntPromotion(Options.get("HonorIntPromotion", false)) {}

void UseStdBitCheck::registerMatchers(MatchFinder *Finder) {
  const auto MakeBinaryOperatorMatcher = [](auto Op) {
    return [=](const auto &LHS, const auto &RHS) {
      return binaryOperator(hasOperatorName(Op),
                            hasLHS(ignoringParenImpCasts(LHS)),
                            hasRHS(ignoringParenImpCasts(RHS)));
    };
  };
  const auto MakeCommutativeBinaryOperatorMatcher = [](auto Op) {
    return [=](const auto &LHS, const auto &RHS) {
      return binaryOperator(
          hasOperatorName(Op),
          hasOperands(ignoringParenImpCasts(LHS), ignoringParenImpCasts(RHS)));
    };
  };

  const auto LogicalAnd = MakeCommutativeBinaryOperatorMatcher("&&");
  const auto Sub = MakeBinaryOperatorMatcher("-");
  const auto ShiftLeft = MakeBinaryOperatorMatcher("<<");
  const auto ShiftRight = MakeBinaryOperatorMatcher(">>");
  const auto BitwiseAnd = MakeCommutativeBinaryOperatorMatcher("&");
  const auto BitwiseOr = MakeCommutativeBinaryOperatorMatcher("|");
  const auto CmpNot = MakeCommutativeBinaryOperatorMatcher("!=");
  const auto CmpGt = MakeBinaryOperatorMatcher(">");
  const auto CmpGte = MakeBinaryOperatorMatcher(">=");
  const auto CmpLt = MakeBinaryOperatorMatcher("<");
  const auto CmpLte = MakeBinaryOperatorMatcher("<=");

  const auto Literal0 = integerLiteral(equals(0));
  const auto Literal1 = integerLiteral(equals(1));

  const auto LogicalNot = [](const auto &Expr) {
    return unaryOperator(hasOperatorName("!"),
                         hasUnaryOperand(ignoringParenImpCasts(Expr)));
  };

  const auto IsNonNull = [=](const auto &Expr) {
    return anyOf(Expr, CmpNot(Expr, Literal0), CmpGt(Expr, Literal0),
                 CmpGte(Expr, Literal1), CmpLt(Literal0, Expr),
                 CmpLte(Literal1, Expr));
  };
  const auto BindDeclRef = [](StringRef Name) {
    return declRefExpr(
        to(varDecl(hasType(isUnsignedInteger())).bind(Name.str())));
  };
  const auto BoundDeclRef = [](StringRef Name) {
    return declRefExpr(to(varDecl(equalsBoundNode(Name.str()))));
  };

  // Determining if an integer is a power of 2 with following pattern:
  // has_one_bit(v) = v && !(v & (v - 1));
  Finder->addMatcher(
      LogicalAnd(IsNonNull(BindDeclRef("v")),
                 LogicalNot(BitwiseAnd(
                     BoundDeclRef("v"),
                     Sub(BoundDeclRef("v"), integerLiteral(equals(1))))))
          .bind("has_one_bit_expr"),
      this);

  // Computing popcount with following pattern:
  // std::bitset<N>(val).count()
  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(0),
          callee(cxxMethodDecl(
              hasName("count"),
              ofClass(cxxRecordDecl(hasName("bitset"), isInStdNamespace())))),
          on(cxxConstructExpr(
              hasArgument(0, expr(hasType(isUnsignedInteger())).bind("v")))))
          .bind("popcount_expr"),
      this);

  // Rotating an integer by a fixed amount
  Finder->addMatcher(
      expr(BitwiseOr(ShiftLeft(BindDeclRef("v"),
                               integerLiteral().bind("shift_left_amount")),
                     ShiftRight(BoundDeclRef("v"),
                                integerLiteral().bind("shift_right_amount"))),
           optionally(hasParent(castExpr(hasType(isInteger())).bind("cast"))))
          .bind("rotate_expr"),
      this);
}

void UseStdBitCheck::registerPPCallbacks(const SourceManager &SM,
                                         Preprocessor *PP,
                                         Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseStdBitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "HonorIntPromotion", HonorIntPromotion);
}

void UseStdBitCheck::check(const MatchFinder::MatchResult &Result) {
  ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();

  if (const auto *MatchedExpr =
          Result.Nodes.getNodeAs<BinaryOperator>("has_one_bit_expr")) {
    const auto *MatchedVarDecl = Result.Nodes.getNodeAs<VarDecl>("v");

    auto Diag =
        diag(MatchedExpr->getBeginLoc(), "use 'std::has_one_bit' instead");
    if (auto R = MatchedExpr->getSourceRange();
        !R.getBegin().isMacroID() && !R.getEnd().isMacroID()) {
      Diag << FixItHint::CreateReplacement(
                  MatchedExpr->getSourceRange(),
                  ("std::has_one_bit(" + MatchedVarDecl->getName() + ")").str())
           << IncludeInserter.createIncludeInsertion(
                  Source.getFileID(MatchedExpr->getBeginLoc()), "<bit>");
    }
  } else if (const auto *MatchedExpr =
                 Result.Nodes.getNodeAs<CXXMemberCallExpr>("popcount_expr")) {
    const auto *BitsetInstantiatedDecl =
        cast<ClassTemplateSpecializationDecl>(MatchedExpr->getRecordDecl());
    const llvm::APSInt BitsetSize =
        BitsetInstantiatedDecl->getTemplateArgs()[0].getAsIntegral();
    const auto *MatchedArg = Result.Nodes.getNodeAs<Expr>("v");
    const uint64_t MatchedVarSize = Context.getTypeSize(MatchedArg->getType());
    if (BitsetSize < MatchedVarSize)
      return;
    auto Diag = diag(MatchedExpr->getBeginLoc(), "use 'std::popcount' instead");
    if (auto R = MatchedExpr->getSourceRange();
        !R.getBegin().isMacroID() && !R.getEnd().isMacroID()) {
      Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                  MatchedArg->getEndLoc().getLocWithOffset(1),
                  MatchedExpr->getRParenLoc().getLocWithOffset(-1)))
           << FixItHint::CreateReplacement(
                  CharSourceRange::getTokenRange(
                      MatchedExpr->getBeginLoc(),
                      MatchedArg->getBeginLoc().getLocWithOffset(-1)),
                  "std::popcount(")
           << IncludeInserter.createIncludeInsertion(
                  Source.getFileID(MatchedExpr->getBeginLoc()), "<bit>");
    }
  } else if (const auto *MatchedExpr =
                 Result.Nodes.getNodeAs<Expr>("rotate_expr")) {
    // Detect if the expression is an explicit cast. If that's the case we don't
    // need to insert a cast.

    bool HasExplicitIntegerCast = false;
    if (const Expr *CE = Result.Nodes.getNodeAs<CastExpr>("cast"))
      HasExplicitIntegerCast = !isa<ImplicitCastExpr>(CE);

    const auto *MatchedVarDecl = Result.Nodes.getNodeAs<VarDecl>("v");
    const llvm::APInt ShiftLeftAmount =
        Result.Nodes.getNodeAs<IntegerLiteral>("shift_left_amount")->getValue();
    const llvm::APInt ShiftRightAmount =
        Result.Nodes.getNodeAs<IntegerLiteral>("shift_right_amount")
            ->getValue();
    const uint64_t MatchedVarSize =
        Context.getTypeSize(MatchedVarDecl->getType());

    // Overflowing shifts
    if (ShiftLeftAmount.sge(MatchedVarSize))
      return;
    if (ShiftRightAmount.sge(MatchedVarSize))
      return;
    // Not a rotation.
    if (MatchedVarSize != (ShiftLeftAmount + ShiftRightAmount))
      return;

    // Only insert cast if the operand is not subject to cast and
    // some implicit promotion happened.
    const bool NeedsIntCast =
        HonorIntPromotion && !HasExplicitIntegerCast &&
        Context.getTypeSize(MatchedExpr->getType()) > MatchedVarSize;
    const bool IsRotl = ShiftRightAmount.sge(ShiftLeftAmount);

    const StringRef ReplacementFuncName = IsRotl ? "rotl" : "rotr";
    const uint64_t ReplacementShiftAmount =
        (IsRotl ? ShiftLeftAmount : ShiftRightAmount).getZExtValue();
    auto Diag = diag(MatchedExpr->getBeginLoc(), "use 'std::%0' instead")
                << ReplacementFuncName;
    if (auto R = MatchedExpr->getSourceRange();
        R.getBegin().isMacroID() || R.getEnd().isMacroID())
      return;

    Diag << FixItHint::CreateReplacement(
                MatchedExpr->getSourceRange(),
                llvm::formatv("{3}std::{0}({1}, {2}){4}", ReplacementFuncName,
                              MatchedVarDecl->getName(), ReplacementShiftAmount,
                              NeedsIntCast ? "static_cast<int>(" : "",
                              NeedsIntCast ? ")" : "")
                    .str())
         << IncludeInserter.createIncludeInsertion(
                Source.getFileID(MatchedExpr->getBeginLoc()), "<bit>");

  } else {
    llvm_unreachable("unexpected match");
  }
}

} // namespace clang::tidy::modernize
