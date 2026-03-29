//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantCastingCheck.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {
AST_MATCHER(Expr, isMacroID) { return Node.getExprLoc().isMacroID(); }
AST_MATCHER_P(OverloadExpr, hasAnyUnresolvedName, ArrayRef<StringRef>, Names) {
  auto DeclName = Node.getName();
  if (!DeclName.isIdentifier())
    return false;
  const IdentifierInfo *II = DeclName.getAsIdentifierInfo();
  return llvm::any_of(Names, [II](StringRef Name) { return II->isStr(Name); });
}
} // namespace

static constexpr StringRef FunctionNames[] = {
    "cast",     "cast_or_null",     "cast_if_present",
    "dyn_cast", "dyn_cast_or_null", "dyn_cast_if_present"};

void RedundantCastingCheck::registerMatchers(MatchFinder *Finder) {
  auto AnyCalleeName = [](ArrayRef<StringRef> CalleeName) {
    return allOf(unless(isMacroID()), unless(cxxMemberCallExpr()),
                 callee(expr(ignoringImpCasts(
                     declRefExpr(to(namedDecl(hasAnyName(CalleeName))),
                                 hasAnyTemplateArgumentLoc(anything()))
                         .bind("callee")))));
  };
  auto AnyCalleeNameInUninstantiatedTemplate =
      [](ArrayRef<StringRef> CalleeName) {
        return allOf(unless(isMacroID()), unless(cxxMemberCallExpr()),
                     callee(expr(ignoringImpCasts(
                         unresolvedLookupExpr(hasAnyUnresolvedName(CalleeName))
                             .bind("callee")))));
      };
  Finder->addMatcher(callExpr(AnyCalleeName(FunctionNames)).bind("call"), this);
  Finder->addMatcher(
      callExpr(AnyCalleeNameInUninstantiatedTemplate(FunctionNames))
          .bind("call"),
      this);
}

static QualType stripPointerOrReference(QualType Ty) {
  QualType Pointee = Ty->getPointeeType();
  if (Pointee.isNull())
    return Ty;
  return Pointee;
}

void RedundantCastingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &Nodes = Result.Nodes;
  const auto *Call = Nodes.getNodeAs<CallExpr>("call");
  if (Call->getNumArgs() != 1)
    return;

  CanQualType RetTy;
  std::string FuncName;
  if (const auto *ResolvedCallee = Nodes.getNodeAs<DeclRefExpr>("callee")) {
    const auto *F = cast<FunctionDecl>(ResolvedCallee->getDecl());
    RetTy = stripPointerOrReference(F->getReturnType())
                ->getCanonicalTypeUnqualified();
    FuncName = F->getName();
  } else if (const auto *UnresolvedCallee =
                 Nodes.getNodeAs<UnresolvedLookupExpr>("callee")) {
    if (UnresolvedCallee->getNumTemplateArgs() != 1)
      return;
    auto TArg = UnresolvedCallee->template_arguments()[0].getArgument();
    if (TArg.getKind() != TemplateArgument::Type)
      return;

    RetTy = TArg.getAsType()->getCanonicalTypeUnqualified();
    FuncName = UnresolvedCallee->getName().getAsString();
  } else {
    llvm_unreachable("");
  }

  const auto *Arg = Call->getArg(0);
  const CanQualType FromTy =
      stripPointerOrReference(Arg->getType())->getCanonicalTypeUnqualified();
  const auto *FromDecl = FromTy->getAsCXXRecordDecl();
  const auto *RetDecl = RetTy->getAsCXXRecordDecl();
  const bool IsDerived =
      FromDecl && RetDecl && FromDecl->isDerivedFrom(RetDecl);
  if (FromTy != RetTy && !IsDerived)
    return;

  auto GetText = [&](SourceRange R) {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(R),
                                *Result.SourceManager, getLangOpts());
  };
  StringRef ArgText = GetText(Arg->getSourceRange());
  diag(Call->getExprLoc(), "redundant use of '%0'")
      << FuncName
      << FixItHint::CreateReplacement(Call->getSourceRange(), ArgText);
  diag(Arg->getExprLoc(), "source expression has type %0", DiagnosticIDs::Note)
      << Arg->getSourceRange() << Arg->IgnoreParenImpCasts()->getType();

  if (FromTy != RetTy) {
    diag(Arg->getExprLoc(), "%0 is a subtype of %1", DiagnosticIDs::Note)
        << FromTy << RetTy;
  }
}

} // namespace clang::tidy::llvm_check
