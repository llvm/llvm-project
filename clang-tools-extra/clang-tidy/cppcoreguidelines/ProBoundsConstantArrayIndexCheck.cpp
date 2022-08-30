//===--- ProBoundsConstantArrayIndexCheck.cpp - clang-tidy-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProBoundsConstantArrayIndexCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

ProBoundsConstantArrayIndexCheck::ProBoundsConstantArrayIndexCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), GslHeader(Options.get("GslHeader", "")),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void ProBoundsConstantArrayIndexCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "GslHeader", GslHeader);
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void ProBoundsConstantArrayIndexCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ProBoundsConstantArrayIndexCheck::registerMatchers(MatchFinder *Finder) {
  // Note: if a struct contains an array member, the compiler-generated
  // constructor has an arraySubscriptExpr.
  Finder->addMatcher(arraySubscriptExpr(hasBase(ignoringImpCasts(hasType(
                                            constantArrayType().bind("type")))),
                                        hasIndex(expr().bind("index")),
                                        unless(hasAncestor(decl(isImplicit()))))
                         .bind("expr"),
                     this);

  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("[]"),
          hasArgument(
              0, hasType(cxxRecordDecl(hasName("::std::array")).bind("type"))),
          hasArgument(1, expr().bind("index")))
          .bind("expr"),
      this);
}

void ProBoundsConstantArrayIndexCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Matched = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *IndexExpr = Result.Nodes.getNodeAs<Expr>("index");

  // This expression can only appear inside ArrayInitLoopExpr, which
  // is always implicitly generated. ArrayInitIndexExpr is not a
  // constant, but we shouldn't report a warning for it.
  if (isa<ArrayInitIndexExpr>(IndexExpr))
    return;

  if (IndexExpr->isValueDependent())
    return; // We check in the specialization.

  Optional<llvm::APSInt> Index =
      IndexExpr->getIntegerConstantExpr(*Result.Context);
  if (!Index) {
    SourceRange BaseRange;
    if (const auto *ArraySubscriptE = dyn_cast<ArraySubscriptExpr>(Matched))
      BaseRange = ArraySubscriptE->getBase()->getSourceRange();
    else
      BaseRange =
          cast<CXXOperatorCallExpr>(Matched)->getArg(0)->getSourceRange();
    SourceRange IndexRange = IndexExpr->getSourceRange();

    auto Diag = diag(Matched->getExprLoc(),
                     "do not use array subscript when the index is "
                     "not an integer constant expression");
    if (!GslHeader.empty()) {
      Diag << FixItHint::CreateInsertion(BaseRange.getBegin(), "gsl::at(")
           << FixItHint::CreateReplacement(
                  SourceRange(BaseRange.getEnd().getLocWithOffset(1),
                              IndexRange.getBegin().getLocWithOffset(-1)),
                  ", ")
           << FixItHint::CreateReplacement(Matched->getEndLoc(), ")")
           << Inserter.createMainFileIncludeInsertion(GslHeader);
    }
    return;
  }

  const auto *StdArrayDecl =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("type");

  // For static arrays, this is handled in clang-diagnostic-array-bounds.
  if (!StdArrayDecl)
    return;

  if (Index->isSigned() && Index->isNegative()) {
    diag(Matched->getExprLoc(), "std::array<> index %0 is negative")
        << toString(*Index, 10);
    return;
  }

  const TemplateArgumentList &TemplateArgs = StdArrayDecl->getTemplateArgs();
  if (TemplateArgs.size() < 2)
    return;
  // First template arg of std::array is the type, second arg is the size.
  const auto &SizeArg = TemplateArgs[1];
  if (SizeArg.getKind() != TemplateArgument::Integral)
    return;
  llvm::APInt ArraySize = SizeArg.getAsIntegral();

  // Get uint64_t values, because different bitwidths would lead to an assertion
  // in APInt::uge.
  if (Index->getZExtValue() >= ArraySize.getZExtValue()) {
    diag(Matched->getExprLoc(),
         "std::array<> index %0 is past the end of the array "
         "(which contains %1 elements)")
        << toString(*Index, 10) << toString(ArraySize, 10, false);
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
