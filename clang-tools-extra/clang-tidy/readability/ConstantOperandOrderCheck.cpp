//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstantOperandOrderCheck.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

/// Out-of-line ctor so vtable is emitted.
ConstantOperandOrderCheck::ConstantOperandOrderCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  // Read options (StringRef -> std::string).
  PreferredSide = Options.get(PreferredSideOption, "Right").str();

  // Parse BinaryOperators option (comma-separated).
  std::string OpsCSV =
      Options.get(BinaryOperatorsOption, "==,!=,<,<=,>,>=").str();

  llvm::SmallVector<llvm::StringRef, 8> Tokens;
  llvm::StringRef(OpsCSV).split(Tokens, ',');
  Operators.clear();
  for (auto Tok : Tokens) {
    llvm::StringRef Trim = Tok.trim();
    if (!Trim.empty())
      Operators.emplace_back(Trim.str());
  }
}

ConstantOperandOrderCheck::~ConstantOperandOrderCheck() = default;

void ConstantOperandOrderCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, PreferredSideOption, PreferredSide);
  Options.store(Opts, BinaryOperatorsOption,
                llvm::join(Operators.begin(), Operators.end(), ","));
}

// ------------------------ helpers ------------------------

namespace {

static const Expr *strip(const Expr *E) {
  return E ? E->IgnoreParenImpCasts() : nullptr;
}

static bool isSimpleConstantExpr(const Expr *E) {
  E = strip(E);
  if (!E)
    return false;

  if (isa<IntegerLiteral>(E) || isa<FloatingLiteral>(E) ||
      isa<StringLiteral>(E) || isa<CharacterLiteral>(E) ||
      isa<CXXBoolLiteralExpr>(E) || isa<CXXNullPtrLiteralExpr>(E) ||
      isa<GNUNullExpr>(E))
    return true;

  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (isa<EnumConstantDecl>(DRE->getDecl()))
      return true;
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD->isConstexpr() || VD->getType().isConstQualified();
  }

  return false;
}

static bool hasSideEffectsExpr(const Expr *E, ASTContext &Ctx) {
  E = strip(E);
  return E && E->HasSideEffects(Ctx);
}

static std::string invertOperatorText(llvm::StringRef Op) {
  if (Op == "<")
    return ">";
  if (Op == ">")
    return "<";
  if (Op == "<=")
    return ">=";
  if (Op == ">=")
    return "<=";
  // symmetric: ==, !=
  return Op.str();
}

} // namespace

// ------------------------ matchers ------------------------

void ConstantOperandOrderCheck::registerMatchers(MatchFinder *Finder) {
  if (Operators.empty())
    return;

  for (const auto &Op : Operators) {
    Finder->addMatcher(binaryOperator(hasOperatorName(Op),
                                      hasLHS(expr().bind("lhs")),
                                      hasRHS(expr().bind("rhs")))
                           .bind("binop"),
                       this);
  }
}

// ------------------------ check / fixit ------------------------

void ConstantOperandOrderCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Bin = Result.Nodes.getNodeAs<BinaryOperator>("binop");
  const auto *L = Result.Nodes.getNodeAs<Expr>("lhs");
  const auto *R = Result.Nodes.getNodeAs<Expr>("rhs");
  if (!Bin || !L || !R)
    return;

  const ASTContext &Ctx = *Result.Context;
  SourceManager &SM = *Result.SourceManager;

  const Expr *LCore = strip(L);
  const Expr *RCore = strip(R);
  const bool LIsConst = isSimpleConstantExpr(LCore);
  const bool RIsConst = isSimpleConstantExpr(RCore);

  // Only when exactly one side is constant.
  if (LIsConst == RIsConst)
    return;

  const bool PreferRight = (PreferredSide == "Right");

  // If it's already on the preferred side -> nothing to do.
  if ((PreferRight && RIsConst && !LIsConst) ||
      (!PreferRight && LIsConst && !RIsConst))
    return;

  // At this point: exactly one side is constant, and it's on the *wrong* side.
  // Emit diagnosis (tests expect a warning even when we won't provide a fix).
  auto D =
      diag(Bin->getOperatorLoc(), "constant operand should be on the %0 side")
      << PreferredSide;

  // Conservative: don't offer fix-its if swapping would move side-effects or if
  // we're inside a macro expansion.
  const bool LSE = L->HasSideEffects(Ctx);
  const bool RSE = R->HasSideEffects(Ctx);
  const bool AnyMacro = L->getBeginLoc().isMacroID() ||
                        R->getBeginLoc().isMacroID() ||
                        Bin->getOperatorLoc().isMacroID();
  if (LSE || RSE || AnyMacro)
    return; // warning-only: no FixIt attached.

  // Get token ranges for the two operands.
  CharSourceRange LRange = CharSourceRange::getTokenRange(L->getSourceRange());
  CharSourceRange RRange = CharSourceRange::getTokenRange(R->getSourceRange());
  if (LRange.isInvalid() || RRange.isInvalid())
    return;

  llvm::StringRef LText = Lexer::getSourceText(LRange, SM, Ctx.getLangOpts());
  llvm::StringRef RText = Lexer::getSourceText(RRange, SM, Ctx.getLangOpts());
  if (LText.empty() || RText.empty())
    return;

  // Compute operator replacement (invert for asymmetric operators).
  const StringRef OpName = Bin->getOpcodeStr();
  const std::string NewOp = invertOperatorText(OpName);

  // Apply operand swaps as two independent replacements (safer than replacing
  // the whole Bin range).
  // Replace left operand with right text:
  D << FixItHint::CreateReplacement(LRange, RText.str());
  // Replace right operand with left text:
  D << FixItHint::CreateReplacement(RRange, LText.str());

  // If needed, replace the operator token too.
  if (NewOp != OpName.str()) {
    // Compute an operator token range robustly: operator start and end.
    SourceLocation OpStart = Bin->getOperatorLoc();
    SourceLocation OpEnd =
        Lexer::getLocForEndOfToken(OpStart, 0, SM, Ctx.getLangOpts());
    if (OpStart.isValid() && OpEnd.isValid()) {
      SourceRange OpRange(OpStart, OpEnd);
      CharSourceRange OpTok = CharSourceRange::getTokenRange(OpRange);
      if (!OpTok.isInvalid())
        D << FixItHint::CreateReplacement(OpTok, NewOp);
    }
  }
}

} // namespace clang::tidy::readability
