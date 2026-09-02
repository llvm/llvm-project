//===--- CallGuardInitCheck.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallGuardInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::pybind {

namespace {

bool isInPybind11Namespace(const Decl *D) {
  if (!D)
    return false;
  for (const DeclContext *DC = D->getDeclContext(); DC; DC = DC->getParent()) {
    if (const auto *ND = dyn_cast<NamespaceDecl>(DC)) {
      if (ND->getName() == "pybind11")
        return true;
    }
  }
  return false;
}

bool isGilScopedRelease(QualType QT) {
  if (QT.isNull())
    return false;
  if (const CXXRecordDecl *RD = QT.getCanonicalType()->getAsCXXRecordDecl()) {
    if (RD->getName() == "gil_scoped_release" && isInPybind11Namespace(RD))
      return true;
  }
  return false;
}

bool isCallGuardWithGilScopedRelease(const Expr *E) {
  if (!E)
    return false;
  E = E->IgnoreParenImpCasts();

  QualType QT = E->getType();
  if (QT.isNull())
    return false;

  const CXXRecordDecl *RD = QT.getCanonicalType()->getAsCXXRecordDecl();
  if (!RD)
    return false;

  if (RD->getName() != "call_guard" || !isInPybind11Namespace(RD))
    return false;

  if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
    const TemplateArgumentList &Args = CTSD->getTemplateArgs();
    for (unsigned I = 0; I < Args.size(); ++I) {
      const TemplateArgument &Arg = Args[I];
      if (Arg.getKind() == TemplateArgument::Type &&
          isGilScopedRelease(Arg.getAsType()))
        return true;
      if (Arg.getKind() == TemplateArgument::Pack) {
        for (const auto &PackArg : Arg.pack_elements()) {
          if (PackArg.getKind() == TemplateArgument::Type &&
              isGilScopedRelease(PackArg.getAsType()))
            return true;
        }
      }
    }
  }

  return false;
}

bool isPybindInit(const Expr *E) {
  if (!E)
    return false;
  E = E->IgnoreParenImpCasts();

  if (const auto *CE = dyn_cast<CallExpr>(E)) {
    if (const FunctionDecl *FD = CE->getDirectCallee()) {
      if (FD->getName() == "init" && isInPybind11Namespace(FD))
        return true;
    }
  }

  QualType QT = E->getType();
  if (!QT.isNull()) {
    if (const CXXRecordDecl *RD = QT.getCanonicalType()->getAsCXXRecordDecl()) {
      StringRef Name = RD->getName();
      if ((Name == "init" || Name == "constructor" || Name == "factory" ||
           Name == "init_factory") &&
          isInPybind11Namespace(RD))
        return true;
    }
  }

  return false;
}

} // namespace

void CallGuardInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("def")))).bind("def_call"), this);
}

void CallGuardInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *DefCall = Result.Nodes.getNodeAs<CallExpr>("def_call");
  if (!DefCall)
    return;

  const Expr *InitArg = nullptr;
  const Expr *CallGuardArg = nullptr;

  for (const Expr *Arg : DefCall->arguments()) {
    const Expr *Unwrapped = Arg->IgnoreParenImpCasts();
    if (!InitArg && isPybindInit(Unwrapped))
      InitArg = Unwrapped;
    if (!CallGuardArg && isCallGuardWithGilScopedRelease(Unwrapped))
      CallGuardArg = Unwrapped;
  }

  if (InitArg && CallGuardArg) {
    diag(CallGuardArg->getExprLoc(),
         "do not use 'py::call_guard<py::gil_scoped_release>' on 'py::init'; "
         "release the GIL inside the factory function body instead");
  }
}

} // namespace clang::tidy::pybind
