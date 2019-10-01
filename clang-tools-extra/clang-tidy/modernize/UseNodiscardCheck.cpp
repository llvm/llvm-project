//===--- UseNodiscardCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseNodiscardCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static bool doesNoDiscardMacroExist(ASTContext &Context,
                                    const llvm::StringRef &MacroId) {
  // Don't check for the Macro existence if we are using an attribute
  // either a C++17 standard attribute or pre C++17 syntax
  if (MacroId.startswith("[[") || MacroId.startswith("__attribute__"))
    return true;

  // Otherwise look up the macro name in the context to see if its defined.
  return Context.Idents.get(MacroId).hasMacroDefinition();
}

namespace {
AST_MATCHER(CXXMethodDecl, isOverloadedOperator) {
  // Don't put ``[[nodiscard]]`` in front of operators.
  return Node.isOverloadedOperator();
}
AST_MATCHER(CXXMethodDecl, isConversionOperator) {
  // Don't put ``[[nodiscard]]`` in front of a conversion decl
  // like operator bool().
  return isa<CXXConversionDecl>(Node);
}
AST_MATCHER(CXXMethodDecl, hasClassMutableFields) {
  // Don't put ``[[nodiscard]]`` on functions on classes with
  // mutable member variables.
  return Node.getParent()->hasMutableFields();
}
AST_MATCHER(ParmVarDecl, hasParameterPack) {
  // Don't put ``[[nodiscard]]`` on functions with parameter pack arguments.
  return Node.isParameterPack();
}
AST_MATCHER(CXXMethodDecl, hasTemplateReturnType) {
  // Don't put ``[[nodiscard]]`` in front of functions returning a template
  // type.
  return Node.getReturnType()->isTemplateTypeParmType() ||
         Node.getReturnType()->isInstantiationDependentType();
}
AST_MATCHER(CXXMethodDecl, isDefinitionOrInline) {
  // A function definition, with optional inline but not the declaration.
  return !(Node.isThisDeclarationADefinition() && Node.isOutOfLine());
}
AST_MATCHER(QualType, isInstantiationDependentType) {
  return Node->isInstantiationDependentType();
}
AST_MATCHER(QualType, isNonConstReferenceOrPointer) {
  // If the function has any non-const-reference arguments
  //    bool foo(A &a)
  // or pointer arguments
  //    bool foo(A*)
  // then they may not care about the return value because of passing data
  // via the arguments.
  return (Node->isTemplateTypeParmType() || Node->isPointerType() ||
          (Node->isReferenceType() &&
           !Node.getNonReferenceType().isConstQualified()) ||
          Node->isInstantiationDependentType());
}
} // namespace

UseNodiscardCheck::UseNodiscardCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NoDiscardMacro(Options.get("ReplacementString", "[[nodiscard]]")) {}

void UseNodiscardCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ReplacementString", NoDiscardMacro);
}

void UseNodiscardCheck::registerMatchers(MatchFinder *Finder) {
  // If we use ``[[nodiscard]]`` attribute, we require at least C++17. Use a
  // macro or ``__attribute__`` with pre c++17 compilers by using
  // ReplacementString option.
  if ((NoDiscardMacro == "[[nodiscard]]" && !getLangOpts().CPlusPlus17) ||
      !getLangOpts().CPlusPlus)
    return;

  auto FunctionObj =
      cxxRecordDecl(hasAnyName("::std::function", "::boost::function"));

  // Find all non-void const methods which have not already been marked to
  // warn on unused result.
  Finder->addMatcher(
      cxxMethodDecl(
          allOf(isConst(), isDefinitionOrInline(),
                unless(anyOf(
                    returns(voidType()), isNoReturn(), isOverloadedOperator(),
                    isVariadic(), hasTemplateReturnType(),
                    hasClassMutableFields(), isConversionOperator(),
                    hasAttr(clang::attr::WarnUnusedResult),
                    hasType(isInstantiationDependentType()),
                    hasAnyParameter(anyOf(
                        parmVarDecl(anyOf(hasType(FunctionObj),
                                          hasType(references(FunctionObj)))),
                        hasType(isNonConstReferenceOrPointer()),
                        hasParameterPack()))))))
          .bind("no_discard"),
      this);
}

void UseNodiscardCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("no_discard");
  // Don't make replacements if the location is invalid or in a macro.
  SourceLocation Loc = MatchedDecl->getLocation();
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  SourceLocation RetLoc = MatchedDecl->getInnerLocStart();

  ASTContext &Context = *Result.Context;

  auto Diag = diag(RetLoc, "function %0 should be marked " + NoDiscardMacro)
              << MatchedDecl;

  // Check for the existence of the keyword being used as the ``[[nodiscard]]``.
  if (!doesNoDiscardMacroExist(Context, NoDiscardMacro))
    return;

  // Possible false positives include:
  // 1. A const member function which returns a variable which is ignored
  // but performs some external I/O operation and the return value could be
  // ignored.
  Diag << FixItHint::CreateInsertion(RetLoc, NoDiscardMacro + " ");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
