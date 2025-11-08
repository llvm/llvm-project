//===--- BoolBitwiseOperationCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolBitwiseOperationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <array>
#include <optional>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

static const DynTypedNode *ignoreParensTowardsTheRoot(const DynTypedNode *N,
                                                      ASTContext *AC) {
  if (const auto *S = N->get<Stmt>()) {
    if (isa<ParenExpr>(S)) {
      auto Parents = AC->getParents(*S);
      for (const auto &Parent : Parents) {
        return ignoreParensTowardsTheRoot(&Parent, AC);
      }
    }
  }
  return N;
}

static bool assignsToBoolean(const BinaryOperator *BinOp, ASTContext *AC) {
  TraversalKindScope RAII(*AC, TK_AsIs);
  auto Parents = AC->getParents(*BinOp);

  for (const auto &Parent : Parents) {
    const auto *ParentNoParen = ignoreParensTowardsTheRoot(&Parent, AC);
    // Special handling for `template<bool bb=true|1>` cases
    if (const auto *D = ParentNoParen->get<Decl>()) {
      if (const auto *NTTPD = dyn_cast<NonTypeTemplateParmDecl>(D)) {
        if (NTTPD->getType().getDesugaredType(*AC)->isBooleanType())
          return true;
      }
    }

    if (const auto *S = ParentNoParen->get<Stmt>()) {
      if (const auto *ICE = dyn_cast<ImplicitCastExpr>(S)) {
        if (ICE->getType().getDesugaredType(*AC)->isBooleanType())
          return true;
      }
    }
  }

  return false;
}

constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 8U>
    OperatorsTransformation{{{"|", "||"},
                             {"|=", "||"},
                             {"&", "&&"},
                             {"&=", "&&"},
                             {"bitand", "and"},
                             {"and_eq", "and"},
                             {"bitor", "or"},
                             {"or_eq", "or"}}};

static llvm::StringRef translate(llvm::StringRef Value) {
  for (const auto &[Bitwise, Logical] : OperatorsTransformation) {
    if (Value == Bitwise)
      return Logical;
  }

  return {};
}

static bool isBooleanBitwise(const BinaryOperator *BinOp, ASTContext *AC,
                             std::optional<bool> &RootAssignsToBoolean);

static bool recheckIsBooleanDeeply(const BinaryOperator *BinOp, ASTContext *AC,
                                   bool &IsBooleanLHS, bool &IsBooleanRHS) {
  std::optional<bool> DummyFlag = false;
  IsBooleanLHS = IsBooleanLHS ||
                 isBooleanBitwise(dyn_cast<BinaryOperator>(
                                      BinOp->getLHS()->IgnoreParenImpCasts()),
                                  AC, DummyFlag);
  IsBooleanRHS = IsBooleanRHS ||
                 isBooleanBitwise(dyn_cast<BinaryOperator>(
                                      BinOp->getRHS()->IgnoreParenImpCasts()),
                                  AC, DummyFlag);
  return true; // just a formal bool for possibility to be invoked from
               // expression
}

static bool isBooleanBitwise(const BinaryOperator *BinOp, ASTContext *AC,
                             std::optional<bool> &RootAssignsToBoolean) {
  if (!BinOp)
    return false;

  for (const auto &[Bitwise, _] : OperatorsTransformation) {
    if (BinOp->getOpcodeStr() == Bitwise) {
      bool IsBooleanLHS = BinOp->getLHS()
                              ->IgnoreImpCasts()
                              ->getType()
                              .getDesugaredType(*AC)
                              ->isBooleanType();
      bool IsBooleanRHS = BinOp->getRHS()
                              ->IgnoreImpCasts()
                              ->getType()
                              .getDesugaredType(*AC)
                              ->isBooleanType();
      for (int i = 0; i < 2;
           !i++ &&
           recheckIsBooleanDeeply(BinOp, AC, IsBooleanLHS, IsBooleanRHS)) {
        if (IsBooleanLHS && IsBooleanRHS) {
          RootAssignsToBoolean = RootAssignsToBoolean.value_or(false);
          return true;
        }
        if (assignsToBoolean(BinOp, AC) ||
            RootAssignsToBoolean.value_or(false)) {
          RootAssignsToBoolean = RootAssignsToBoolean.value_or(true);
          return true;
        }
        if (BinOp->isCompoundAssignmentOp() && IsBooleanLHS) {
          RootAssignsToBoolean = RootAssignsToBoolean.value_or(true);
          return true;
        }
      }
    }
  }
  return false;
}

static const Expr *getAcceptableCompoundsLHS(const BinaryOperator *BinOp) {
  assert(BinOp->isCompoundAssignmentOp());

  if (const auto *DeclRefLHS =
          dyn_cast<DeclRefExpr>(BinOp->getLHS()->IgnoreImpCasts()))
    return DeclRefLHS;
  else if (const auto *MemberLHS =
               dyn_cast<MemberExpr>(BinOp->getLHS()->IgnoreImpCasts()))
    return MemberLHS;

  return nullptr;
}

BoolBitwiseOperationCheck::BoolBitwiseOperationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", false)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(unless(isExpansionInSystemHeader()),
                     unless(hasParent(binaryOperator())) // ignoring parenExpr
                     )
          .bind("binOpRoot"),
      this);
}

void BoolBitwiseOperationCheck::emitWarningAndChangeOperatorsIfPossible(
    const BinaryOperator *BinOp, const BinaryOperator *ParentBinOp,
    const clang::SourceManager &SM, clang::ASTContext &Ctx) {
  auto DiagEmitter = [BinOp, this] {
    return diag(BinOp->getOperatorLoc(),
                "use logical operator '%0' for boolean semantics instead of "
                "bitwise operator '%1'")
           << translate(BinOp->getOpcodeStr()) << BinOp->getOpcodeStr();
  };

  const bool HasVolatileOperand = llvm::any_of(
      std::array{BinOp->getLHS(), BinOp->getRHS()}, [&](const Expr *E) {
        return E->IgnoreImpCasts()
            ->getType()
            .getDesugaredType(Ctx)
            .isVolatileQualified();
      });
  if (HasVolatileOperand)
    return static_cast<void>(DiagEmitter());

  const bool HasSideEffects = BinOp->getRHS()->HasSideEffects(
      Ctx, /*IncludePossibleEffects=*/!StrictMode);
  if (HasSideEffects)
    return static_cast<void>(DiagEmitter());

  SourceLocation Loc = BinOp->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  const StringRef FixSpelling =
      translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));

  if (FixSpelling.empty())
    return static_cast<void>(DiagEmitter());

  FixItHint InsertEqual;
  if (BinOp->isCompoundAssignmentOp()) {
    const auto *LHS = getAcceptableCompoundsLHS(BinOp);
    if (!LHS)
      return static_cast<void>(DiagEmitter());
    const SourceLocation LocLHS = LHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    const SourceLocation InsertLoc =
        clang::Lexer::getLocForEndOfToken(LocLHS, 0, SM, Ctx.getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    auto SourceText = static_cast<std::string>(Lexer::getSourceText(
        CharSourceRange::getTokenRange(LHS->getSourceRange()), SM,
        Ctx.getLangOpts()));
    llvm::erase_if(SourceText,
                   [](unsigned char ch) { return std::isspace(ch); });
    InsertEqual = FixItHint::CreateInsertion(InsertLoc, " = " + SourceText);
  }

  auto ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  std::optional<BinaryOperatorKind> ParentOpcode;
  if (ParentBinOp)
    ParentOpcode = ParentBinOp->getOpcode();

  const auto *RHS = dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts());
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const Expr *SurroundedExpr = nullptr;
  if ((BinOp->getOpcode() == BO_Or && ParentOpcode == BO_LAnd) ||
      (BinOp->getOpcode() == BO_And &&
       llvm::is_contained({BO_Xor, BO_Or}, ParentOpcode))) {
    const Expr *Side = ParentBinOp->getLHS()->IgnoreParenImpCasts() == BinOp
                           ? ParentBinOp->getLHS()
                           : ParentBinOp->getRHS();
    SurroundedExpr = Side->IgnoreImpCasts();
    assert(SurroundedExpr->IgnoreParens() == BinOp);
  } else if (BinOp->getOpcode() == BO_AndAssign && RHSOpcode == BO_LOr)
    SurroundedExpr = RHS;

  if (SurroundedExpr && isa<ParenExpr>(SurroundedExpr))
    SurroundedExpr = nullptr;

  FixItHint InsertBrace1;
  FixItHint InsertBrace2;
  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, SM, Ctx.getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter() << InsertEqual << ReplaceOperator << InsertBrace1
                << InsertBrace2;
}

void BoolBitwiseOperationCheck::visitBinaryTreesNode(
    const BinaryOperator *BinOp, const BinaryOperator *ParentBinOp,
    const clang::SourceManager &SM, clang::ASTContext &Ctx,
    std::optional<bool> &RootAssignsToBoolean) {
  if (!BinOp)
    return;

  if (isBooleanBitwise(BinOp, &Ctx, RootAssignsToBoolean))
    emitWarningAndChangeOperatorsIfPossible(BinOp, ParentBinOp, SM, Ctx);

  visitBinaryTreesNode(
      dyn_cast<BinaryOperator>(BinOp->getLHS()->IgnoreParenImpCasts()), BinOp,
      SM, Ctx, RootAssignsToBoolean);
  visitBinaryTreesNode(
      dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreParenImpCasts()), BinOp,
      SM, Ctx, RootAssignsToBoolean);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *binOpRoot = Result.Nodes.getNodeAs<BinaryOperator>("binOpRoot");
  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;
  std::optional<bool> RootAssignsToBoolean = std::nullopt;
  visitBinaryTreesNode(binOpRoot, nullptr, SM, Ctx, RootAssignsToBoolean);
}

} // namespace clang::tidy::misc