//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferStaticOverAnonymousNamespaceCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {

AST_MATCHER(NamedDecl, isInMacro) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

AST_MATCHER(VarDecl, isLocalVariable) { return Node.isLocalVarDecl(); }

AST_MATCHER(Decl, isLexicallyInAnonymousNamespace) {
  for (const DeclContext *DC = Node.getLexicalDeclContext(); DC != nullptr;
       DC = DC->getLexicalParent()) {
    if (const auto *ND = dyn_cast<NamespaceDecl>(DC))
      if (ND->isAnonymousNamespace())
        return true;
  }

  return false;
}

} // namespace

PreferStaticOverAnonymousNamespaceCheck::
    PreferStaticOverAnonymousNamespaceCheck(StringRef Name,
                                            ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowVariableDeclarations(Options.get("AllowVariableDeclarations", true)),
      AllowMemberFunctionsInClass(
          Options.get("AllowMemberFunctionsInClass", true)) {}

void PreferStaticOverAnonymousNamespaceCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowVariableDeclarations", AllowVariableDeclarations);
  Options.store(Opts, "AllowMemberFunctionsInClass",
                AllowMemberFunctionsInClass);
}

void PreferStaticOverAnonymousNamespaceCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto IsDefinitionInAnonymousNamespace = allOf(
      unless(isExpansionInSystemHeader()), isLexicallyInAnonymousNamespace(),
      unless(isInMacro()), isDefinition());

  if (AllowMemberFunctionsInClass) {
    Finder->addMatcher(
        functionDecl(IsDefinitionInAnonymousNamespace,
                     unless(anyOf(hasParent(cxxRecordDecl()),
                                  hasParent(functionTemplateDecl(
                                      hasParent(cxxRecordDecl()))))))
            .bind("function"),
        this);
  } else {
    Finder->addMatcher(
        functionDecl(IsDefinitionInAnonymousNamespace).bind("function"), this);
  }

  if (!AllowVariableDeclarations)
    Finder->addMatcher(varDecl(IsDefinitionInAnonymousNamespace,
                               unless(isLocalVariable()), unless(parmVarDecl()))
                           .bind("var"),
                       this);
}

void PreferStaticOverAnonymousNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {

  if (const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("function")) {
    if (Func->isCXXClassMember())
      diag(Func->getLocation(),
           "place definition of method %0 outside of an anonymous namespace")
          << Func;
    else if (Func->isStatic())
      diag(Func->getLocation(),
           "place static function %0 outside of an anonymous namespace")
          << Func;
    else
      diag(Func->getLocation(),
           "function %0 is declared in an anonymous namespace; "
           "prefer using 'static' for restricting visibility")
          << Func;
    return;
  }

  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    if (Var->getStorageClass() == SC_Static)
      diag(Var->getLocation(),
           "place static variable %0 outside of an anonymous namespace")
          << Var;
    else
      diag(Var->getLocation(),
           "variable %0 is declared in an anonymous namespace; "
           "prefer using 'static' for restricting visibility")
          << Var;
  }
}

} // namespace clang::tidy::llvm_check
