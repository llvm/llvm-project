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

static const Stmt* ignoreParenExpr(const Stmt* S) {
  while (const auto* PE = dyn_cast<ParenExpr>(S)) {
    S = PE->getSubExpr();
  }
  return S;
}

template<typename R>
static bool
containsNodeIgnoringParenExpr(R&& children, const Stmt* Node) {
  for (const Stmt* Child : children) {
    if (ignoreParenExpr(Child) == Node) {
      return true;
    }
  }
  return false;
}

static bool isBooleanImplicitCastStmt(const Stmt* S) {
  const auto* ICE = dyn_cast<ImplicitCastExpr>(S);
  return ICE && ICE->getType()->isBooleanType();
}

namespace {
AST_MATCHER(BinaryOperator, assignsToBoolean) {
  auto Parents = Finder->getASTContext().getParents(Node);
  
  for (const auto& Parent : Parents) {
    // Check for Decl parents
    if (const auto* D = Parent.template get<Decl>()) {
      const Expr* InitExpr = nullptr;
      
      if (const auto* VD = dyn_cast<VarDecl>(D)) {
        InitExpr = VD->getInit();
      } else if (const auto* FD = dyn_cast<FieldDecl>(D)) {
        InitExpr = FD->getInClassInitializer();
      } else if (const auto* NTTPD = dyn_cast<NonTypeTemplateParmDecl>(D)) {
        if (NTTPD->getType()->isBooleanType()) {
          return true;
        }
      }
      
      if (InitExpr && isBooleanImplicitCastStmt(InitExpr)) {
        return true;
      }
    }
    
    // Check for Stmt parents
    if (const auto* S = Parent.template get<Stmt>()) {
      for (const Stmt* Child : S->children()) {
        if (isBooleanImplicitCastStmt(Child) && containsNodeIgnoringParenExpr(Child->children(), &Node)) {
          return true;
        }
      }
    }
  }
  
  return false;
}

} // namespace

static const NamedDecl *
getLHSNamedDeclIfCompoundAssign(const BinaryOperator *BO) {
  if (BO->isCompoundAssignmentOp()) {
    const auto *DeclRefLHS =
        dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreImpCasts());
    return DeclRefLHS ? DeclRefLHS->getDecl() : nullptr;
  }
  return nullptr;
}

constexpr std::array<llvm::StringRef, 4U> OperatorsNames{"|", "&", "|=", "&="};

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
      binaryOperator(
          unless(isExpansionInSystemHeader()),
          hasAnyOperatorName(OperatorsNames),
          anyOf(
            hasOperands(expr(hasType(booleanType())), expr(hasType(booleanType()))),
            assignsToBoolean()
          ),
          // isBooleanLikeOperands(),
          optionally(hasParent( // to simple implement transformations like
                                // `a&&b|c` -> `a&&(b||c)`
              binaryOperator().bind("p"))))
          .bind("op"),
      this);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("op");

  auto DiagEmitter = [MatchedExpr, this] {
    const NamedDecl *ND = getLHSNamedDeclIfCompoundAssign(MatchedExpr);
    return diag(MatchedExpr->getOperatorLoc(),
                "use logical operator '%0' for boolean %select{variable "
                "%2|values}1 instead of bitwise operator '%3'")
           << translate(MatchedExpr->getOpcodeStr()) << (ND == nullptr) << ND
           << MatchedExpr->getOpcodeStr();
  };

  const bool HasVolatileOperand = llvm::any_of(
      std::array{MatchedExpr->getLHS(), MatchedExpr->getRHS()},
      [](const Expr *E) {
        return E->IgnoreImpCasts()->getType().isVolatileQualified();
      });
  if (HasVolatileOperand)
    return static_cast<void>(DiagEmitter());

  const bool HasSideEffects = MatchedExpr->getRHS()->HasSideEffects(
      *Result.Context, /*IncludePossibleEffects=*/!StrictMode);
  if (HasSideEffects)
    return static_cast<void>(DiagEmitter());

  SourceLocation Loc = MatchedExpr->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  Loc = Result.SourceManager->getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid())
    return static_cast<void>(IgnoreMacros || DiagEmitter());

  const StringRef FixSpelling = translate(Lexer::getSourceText(
      TokenRange, *Result.SourceManager, Result.Context->getLangOpts()));

  if (FixSpelling.empty())
    return static_cast<void>(DiagEmitter());

  FixItHint InsertEqual;
  if (MatchedExpr->isCompoundAssignmentOp()) {
    const auto *DeclRefLHS =
        dyn_cast<DeclRefExpr>(MatchedExpr->getLHS()->IgnoreImpCasts());
    if (!DeclRefLHS)
      return static_cast<void>(DiagEmitter());
    const SourceLocation LocLHS = DeclRefLHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    const SourceLocation InsertLoc = clang::Lexer::getLocForEndOfToken(
        LocLHS, 0, *Result.SourceManager, Result.Context->getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    InsertEqual = FixItHint::CreateInsertion(
        InsertLoc, " = " + DeclRefLHS->getDecl()->getNameAsString());
  }

  auto ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p");
  std::optional<BinaryOperatorKind> ParentOpcode;
  if (Parent)
    ParentOpcode = Parent->getOpcode();

  const auto *RHS =
      dyn_cast<BinaryOperator>(MatchedExpr->getRHS()->IgnoreImpCasts());
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const Expr *SurroundedExpr = nullptr;
  if ((MatchedExpr->getOpcode() == BO_Or && ParentOpcode == BO_LAnd) ||
      (MatchedExpr->getOpcode() == BO_And &&
       llvm::is_contained({BO_Xor, BO_Or}, ParentOpcode))) {
    const Expr *Side = Parent->getLHS()->IgnoreParenImpCasts() == MatchedExpr
                           ? Parent->getLHS()
                           : Parent->getRHS();
    SurroundedExpr = Side->IgnoreImpCasts();
    assert(SurroundedExpr->IgnoreParens() == MatchedExpr);
  } else if (MatchedExpr->getOpcode() == BO_AndAssign && RHSOpcode == BO_LOr)
    SurroundedExpr = RHS;

  if (SurroundedExpr && isa<ParenExpr>(SurroundedExpr))
    SurroundedExpr = nullptr;

  FixItHint InsertBrace1;
  FixItHint InsertBrace2;
  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, *Result.SourceManager,
        Result.Context->getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID())
      return static_cast<void>(IgnoreMacros || DiagEmitter());
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter() << InsertEqual << ReplaceOperator << InsertBrace1
                << InsertBrace2;
}

} // namespace clang::tidy::misc