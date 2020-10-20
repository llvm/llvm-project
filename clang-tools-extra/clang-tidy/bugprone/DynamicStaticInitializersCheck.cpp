//===--- DynamicStaticInitializersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicStaticInitializersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

AST_MATCHER(clang::VarDecl, hasConstantDeclaration) {
  const Expr *Init = Node.getInit();
  if (Init && !Init->isValueDependent()) {
    if (Node.isConstexpr())
      return true;
    return Node.evaluateValue();
  }
  return false;
}

DynamicStaticInitializersCheck::DynamicStaticInitializersCheck(StringRef Name,
                                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
        "HeaderFileExtensions", utils::defaultHeaderFileExtensions())) {
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    llvm::errs() << "Invalid header file extension: "
                 << RawStringHeaderFileExtensions << "\n";
  }
}

void DynamicStaticInitializersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void DynamicStaticInitializersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(hasGlobalStorage(), unless(hasConstantDeclaration())).bind("var"),
      this);
}

void DynamicStaticInitializersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");
  SourceLocation Loc = Var->getLocation();
  if (!Loc.isValid() || !utils::isPresumedLocInHeaderFile(Loc, *Result.SourceManager,
                                                          HeaderFileExtensions))
    return;
  // If the initializer is a constant expression, then the compiler
  // doesn't have to dynamically initialize it.
  diag(Loc, "static variable %0 may be dynamically initialized in this header file")
    << Var;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
