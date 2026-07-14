//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LambdaCaptureLifetimeCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static const LambdaExpr *getEscapingLambdaFromArgument(const Expr *E) {
  if (!E)
    return nullptr;

  E = E->IgnoreParenImpCasts();

  if (const auto *Lambda = dyn_cast<LambdaExpr>(E))
    return Lambda;

  if (const auto *Cleanups = dyn_cast<ExprWithCleanups>(E))
    return getEscapingLambdaFromArgument(Cleanups->getSubExpr());

  if (const auto *Temporary = dyn_cast<MaterializeTemporaryExpr>(E))
    return getEscapingLambdaFromArgument(Temporary->getSubExpr());

  if (const auto *Temporary = dyn_cast<CXXBindTemporaryExpr>(E))
    return getEscapingLambdaFromArgument(Temporary->getSubExpr());

  if (const auto *Cast = dyn_cast<CastExpr>(E))
    return getEscapingLambdaFromArgument(Cast->getSubExpr());

  if (const auto *Construct = dyn_cast<CXXConstructExpr>(E)) {
    if (Construct->getNumArgs() == 1)
      return getEscapingLambdaFromArgument(Construct->getArg(0));
    return nullptr;
  }

  if (const auto *InitList = dyn_cast<InitListExpr>(E)) {
    if (InitList->getNumInits() == 1)
      return getEscapingLambdaFromArgument(InitList->getInit(0));
    return nullptr;
  }

  return nullptr;
}

static const LambdaExpr *getEscapingLambda(const CXXConstructExpr *Construct) {
  for (const Expr *Arg : Construct->arguments())
    if (const LambdaExpr *Lambda = getEscapingLambdaFromArgument(Arg))
      return Lambda;
  return nullptr;
}

static const LambdaExpr *getEscapingLambda(const CallExpr *Call) {
  for (const Expr *Arg : Call->arguments())
    if (const LambdaExpr *Lambda = getEscapingLambdaFromArgument(Arg))
      return Lambda;
  return nullptr;
}

static bool capturesLocalVariableByReference(const LambdaExpr *Lambda) {
  return llvm::any_of(Lambda->captures(), [](const LambdaCapture &Capture) {
    if (Capture.capturesVariable() && Capture.getCaptureKind() == LCK_ByRef)
      if (const auto *Var = dyn_cast<VarDecl>(Capture.getCapturedVar()))
        return Var->hasLocalStorage();
    return false;
  });
}

LambdaCaptureLifetimeCheck::LambdaCaptureLifetimeCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AsyncClasses(utils::options::parseStringList(
          Options.get("AsyncClasses", "::std::thread;::std::jthread"))),
      AsyncFunctions(utils::options::parseStringList(
          Options.get("AsyncFunctions", "::std::async"))),
      StorageClasses(utils::options::parseStringList(
          Options.get("StorageClasses", "::std::vector"))),
      StorageFunctions(utils::options::parseStringList(Options.get(
          "StorageFunctions", "push_back;emplace_back;insert;assign"))) {}

void LambdaCaptureLifetimeCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AsyncClasses",
                utils::options::serializeStringList(AsyncClasses));
  Options.store(Opts, "AsyncFunctions",
                utils::options::serializeStringList(AsyncFunctions));
  Options.store(Opts, "StorageClasses",
                utils::options::serializeStringList(StorageClasses));
  Options.store(Opts, "StorageFunctions",
                utils::options::serializeStringList(StorageFunctions));
}

void LambdaCaptureLifetimeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                                          ofClass(hasAnyName(AsyncClasses)))),
                                      hasAnyArgument(expr()))
                         .bind("escape-point"),
                     this);

  Finder->addMatcher(callExpr(callee(functionDecl(hasAnyName(AsyncFunctions))),
                              hasAnyArgument(expr()))
                         .bind("escape-point"),
                     this);

  auto LongLivedStorage = anyOf(varDecl(hasGlobalStorage()), fieldDecl());

  auto IsDirectRef = declRefExpr(to(LongLivedStorage));
  auto IsMemberRef = memberExpr(member(LongLivedStorage));
  auto StorageClass = cxxRecordDecl(
      anyOf(hasAnyName(StorageClasses),
            classTemplateSpecializationDecl(hasAnyName(StorageClasses))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasAnyName(StorageFunctions),
                               ofClass(StorageClass))),
          on(expr(ignoringParenImpCasts(anyOf(IsDirectRef, IsMemberRef)))),
          hasAnyArgument(expr()))
          .bind("escape-point"),
      this);
}

void LambdaCaptureLifetimeCheck::check(const MatchFinder::MatchResult &Result) {
  const LambdaExpr *Lambda = nullptr;
  if (const auto *Construct =
          Result.Nodes.getNodeAs<CXXConstructExpr>("escape-point"))
    Lambda = getEscapingLambda(Construct);
  else if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("escape-point"))
    Lambda = getEscapingLambda(Call);

  if (Lambda && capturesLocalVariableByReference(Lambda)) {
    diag(Lambda->getBeginLoc(),
         "lambda captures local variables by reference, but escapes the local "
         "scope, potentially causing a use-after-free");
  }
}

} // namespace clang::tidy::bugprone
