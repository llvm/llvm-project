//===--- ImplicitWideningOfMultiplicationResultCheck.cpp - clang-tidy -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplicitWideningOfMultiplicationResultCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang {
namespace {
AST_MATCHER(ImplicitCastExpr, isPartOfExplicitCast) {
  return Node.isPartOfExplicitCast();
}
} // namespace
} // namespace clang

namespace clang::tidy::bugprone {

static const Expr *getLHSOfMulBinOp(const Expr *E) {
  assert(E == E->IgnoreParens() && "Already skipped all parens!");
  // Is this:  long r = int(x) * int(y);  ?
  // FIXME: shall we skip brackets/casts/etc?
  const auto *BO = dyn_cast<BinaryOperator>(E);
  if (!BO || BO->getOpcode() != BO_Mul)
    // FIXME: what about:  long r = int(x) + (int(y) * int(z));  ?
    return nullptr;
  return BO->getLHS()->IgnoreParens();
}

ImplicitWideningOfMultiplicationResultCheck::
    ImplicitWideningOfMultiplicationResultCheck(StringRef Name,
                                                ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UseCXXStaticCastsInCppSources(
          Options.get("UseCXXStaticCastsInCppSources", true)),
      UseCXXHeadersInCppSources(Options.get("UseCXXHeadersInCppSources", true)),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void ImplicitWideningOfMultiplicationResultCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void ImplicitWideningOfMultiplicationResultCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseCXXStaticCastsInCppSources",
                UseCXXStaticCastsInCppSources);
  Options.store(Opts, "UseCXXHeadersInCppSources", UseCXXHeadersInCppSources);
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

std::optional<FixItHint>
ImplicitWideningOfMultiplicationResultCheck::includeStddefHeader(
    SourceLocation File) {
  return IncludeInserter.createIncludeInsertion(
      Result->SourceManager->getFileID(File),
      ShouldUseCXXHeader ? "<cstddef>" : "<stddef.h>");
}

void ImplicitWideningOfMultiplicationResultCheck::handleImplicitCastExpr(
    const ImplicitCastExpr *ICE) {
  ASTContext *Context = Result->Context;

  const Expr *E = ICE->getSubExpr()->IgnoreParens();
  QualType Ty = ICE->getType();
  QualType ETy = E->getType();

  assert(!ETy->isDependentType() && !Ty->isDependentType() &&
         "Don't expect to ever get here in template Context.");

  // This must be a widening cast. Else we do not care.
  unsigned SrcWidth = Context->getIntWidth(ETy);
  unsigned TgtWidth = Context->getIntWidth(Ty);
  if (TgtWidth <= SrcWidth)
    return;

  // Does the index expression look like it might be unintentionally computed
  // in a narrower-than-wanted type?
  const Expr *LHS = getLHSOfMulBinOp(E);
  if (!LHS)
    return;

  // Ok, looks like we should diagnose this.
  diag(E->getBeginLoc(), "performing an implicit widening conversion to type "
                         "%0 of a multiplication performed in type %1")
      << Ty << E->getType();

  {
    auto Diag = diag(E->getBeginLoc(),
                     "make conversion explicit to silence this warning",
                     DiagnosticIDs::Note)
                << E->getSourceRange();
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        E->getEndLoc(), 0, *Result->SourceManager, getLangOpts());
    if (ShouldUseCXXStaticCast)
      Diag << FixItHint::CreateInsertion(
                  E->getBeginLoc(), "static_cast<" + Ty.getAsString() + ">(")
           << FixItHint::CreateInsertion(EndLoc, ")");
    else
      Diag << FixItHint::CreateInsertion(E->getBeginLoc(),
                                         "(" + Ty.getAsString() + ")(")
           << FixItHint::CreateInsertion(EndLoc, ")");
    Diag << includeStddefHeader(E->getBeginLoc());
  }

  QualType WideExprTy;
  // Get Ty of the same signedness as ExprTy, because we only want to suggest
  // to widen the computation, but not change it's signedness domain.
  if (Ty->isSignedIntegerType() == ETy->isSignedIntegerType())
    WideExprTy = Ty;
  else if (Ty->isSignedIntegerType()) {
    assert(ETy->isUnsignedIntegerType() &&
           "Expected source type to be signed.");
    WideExprTy = Context->getCorrespondingUnsignedType(Ty);
  } else {
    assert(Ty->isUnsignedIntegerType() &&
           "Expected target type to be unsigned.");
    assert(ETy->isSignedIntegerType() &&
           "Expected source type to be unsigned.");
    WideExprTy = Context->getCorrespondingSignedType(Ty);
  }

  {
    auto Diag = diag(E->getBeginLoc(), "perform multiplication in a wider type",
                     DiagnosticIDs::Note)
                << LHS->getSourceRange();

    if (ShouldUseCXXStaticCast)
      Diag << FixItHint::CreateInsertion(LHS->getBeginLoc(),
                                         "static_cast<" +
                                             WideExprTy.getAsString() + ">(")
           << FixItHint::CreateInsertion(
                  Lexer::getLocForEndOfToken(LHS->getEndLoc(), 0,
                                             *Result->SourceManager,
                                             getLangOpts()),
                  ")");
    else
      Diag << FixItHint::CreateInsertion(LHS->getBeginLoc(),
                                         "(" + WideExprTy.getAsString() + ")");
    Diag << includeStddefHeader(LHS->getBeginLoc());
  }
}

void ImplicitWideningOfMultiplicationResultCheck::handlePointerOffsetting(
    const Expr *E) {
  ASTContext *Context = Result->Context;

  // We are looking for a pointer offset operation,
  // with one hand being a pointer, and another one being an offset.
  const Expr *PointerExpr = nullptr, *IndexExpr = nullptr;
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    PointerExpr = BO->getLHS();
    IndexExpr = BO->getRHS();
  } else if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    PointerExpr = ASE->getLHS();
    IndexExpr = ASE->getRHS();
  } else
    return;

  if (IndexExpr->getType()->isPointerType())
    std::swap(PointerExpr, IndexExpr);

  if (!PointerExpr->getType()->isPointerType() ||
      IndexExpr->getType()->isPointerType())
    return;

  IndexExpr = IndexExpr->IgnoreParens();

  QualType IndexExprType = IndexExpr->getType();

  // If the index expression's type is not known (i.e. we are in a template),
  // we can't do anything here.
  if (IndexExprType->isDependentType())
    return;

  QualType SSizeTy = Context->getPointerDiffType();
  QualType USizeTy = Context->getSizeType();
  QualType SizeTy = IndexExprType->isSignedIntegerType() ? SSizeTy : USizeTy;
  // FIXME: is there a way to actually get the QualType for size_t/ptrdiff_t?
  // Note that SizeTy.getAsString() will be unsigned long/..., NOT size_t!
  StringRef TyAsString =
      IndexExprType->isSignedIntegerType() ? "ptrdiff_t" : "size_t";

  // So, is size_t actually wider than the result of the multiplication?
  if (Context->getIntWidth(IndexExprType) >= Context->getIntWidth(SizeTy))
    return;

  // Does the index expression look like it might be unintentionally computed
  // in a narrower-than-wanted type?
  const Expr *LHS = getLHSOfMulBinOp(IndexExpr);
  if (!LHS)
    return;

  // Ok, looks like we should diagnose this.
  diag(E->getBeginLoc(),
       "result of multiplication in type %0 is used as a pointer offset after "
       "an implicit widening conversion to type '%1'")
      << IndexExprType << TyAsString;

  {
    auto Diag = diag(IndexExpr->getBeginLoc(),
                     "make conversion explicit to silence this warning",
                     DiagnosticIDs::Note)
                << IndexExpr->getSourceRange();
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        IndexExpr->getEndLoc(), 0, *Result->SourceManager, getLangOpts());
    if (ShouldUseCXXStaticCast)
      Diag << FixItHint::CreateInsertion(
                  IndexExpr->getBeginLoc(),
                  (Twine("static_cast<") + TyAsString + ">(").str())
           << FixItHint::CreateInsertion(EndLoc, ")");
    else
      Diag << FixItHint::CreateInsertion(IndexExpr->getBeginLoc(),
                                         (Twine("(") + TyAsString + ")(").str())
           << FixItHint::CreateInsertion(EndLoc, ")");
    Diag << includeStddefHeader(IndexExpr->getBeginLoc());
  }

  {
    auto Diag =
        diag(IndexExpr->getBeginLoc(), "perform multiplication in a wider type",
             DiagnosticIDs::Note)
        << LHS->getSourceRange();

    if (ShouldUseCXXStaticCast)
      Diag << FixItHint::CreateInsertion(
                  LHS->getBeginLoc(),
                  (Twine("static_cast<") + TyAsString + ">(").str())
           << FixItHint::CreateInsertion(
                  Lexer::getLocForEndOfToken(IndexExpr->getEndLoc(), 0,
                                             *Result->SourceManager,
                                             getLangOpts()),
                  ")");
    else
      Diag << FixItHint::CreateInsertion(LHS->getBeginLoc(),
                                         (Twine("(") + TyAsString + ")").str());
    Diag << includeStddefHeader(LHS->getBeginLoc());
  }
}

void ImplicitWideningOfMultiplicationResultCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(implicitCastExpr(unless(anyOf(isInTemplateInstantiation(),
                                                   isPartOfExplicitCast())),
                                      hasCastKind(CK_IntegralCast))
                         .bind("x"),
                     this);
  Finder->addMatcher(
      arraySubscriptExpr(unless(isInTemplateInstantiation())).bind("x"), this);
  Finder->addMatcher(binaryOperator(unless(isInTemplateInstantiation()),
                                    hasType(isAnyPointer()),
                                    hasAnyOperatorName("+", "-", "+=", "-="))
                         .bind("x"),
                     this);
}

void ImplicitWideningOfMultiplicationResultCheck::check(
    const MatchFinder::MatchResult &Result) {
  this->Result = &Result;
  ShouldUseCXXStaticCast =
      UseCXXStaticCastsInCppSources && Result.Context->getLangOpts().CPlusPlus;
  ShouldUseCXXHeader =
      UseCXXHeadersInCppSources && Result.Context->getLangOpts().CPlusPlus;

  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<ImplicitCastExpr>("x"))
    handleImplicitCastExpr(MatchedDecl);
  else if (const auto *MatchedDecl =
               Result.Nodes.getNodeAs<ArraySubscriptExpr>("x"))
    handlePointerOffsetting(MatchedDecl);
  else if (const auto *MatchedDecl =
               Result.Nodes.getNodeAs<BinaryOperator>("x"))
    handlePointerOffsetting(MatchedDecl);
}

} // namespace clang::tidy::bugprone
