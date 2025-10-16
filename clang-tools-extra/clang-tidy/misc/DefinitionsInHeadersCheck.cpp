//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefinitionsInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

AST_MATCHER_P(NamedDecl, usesHeaderFileExtension, FileExtensionsSet,
              HeaderFileExtensions) {
  return utils::isExpansionLocInHeaderFile(
      Node.getBeginLoc(), Finder->getASTContext().getSourceManager(),
      HeaderFileExtensions);
}

} // namespace

DefinitionsInHeadersCheck::DefinitionsInHeadersCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HeaderFileExtensions(Context->getHeaderFileExtensions()) {}

void DefinitionsInHeadersCheck::registerMatchers(MatchFinder *Finder) {
  auto DefinitionMatcher =
      anyOf(functionDecl(isDefinition(), unless(isDeleted())),
            varDecl(isDefinition()));
  Finder->addMatcher(namedDecl(DefinitionMatcher,
                               usesHeaderFileExtension(HeaderFileExtensions))
                         .bind("name-decl"),
                     this);
}

void DefinitionsInHeadersCheck::check(const MatchFinder::MatchResult &Result) {
  // Don't run the check in failing TUs.
  if (Result.Context->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  // C++ [basic.def.odr] p6:
  // There can be more than one definition of a class type, enumeration type,
  // inline function with external linkage, class template, non-static function
  // template, static data member of a class template, member function of a
  // class template, or template specialization for which some template
  // parameters are not specifiedin a program provided that each definition
  // appears in a different translation unit, and provided the definitions
  // satisfy the following requirements.
  const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("name-decl");
  assert(ND);
  if (ND->isInvalidDecl())
    return;

  // Internal linkage variable definitions are ignored for now:
  //   const int a = 1;
  //   static int b = 1;
  //   namespace { int c = 1; }
  //
  // Although these might also cause ODR violations, we can be less certain and
  // should try to keep the false-positive rate down.
  if (!ND->hasExternalFormalLinkage() || ND->isInAnonymousNamespace())
    return;

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    // Inline functions are allowed.
    if (FD->isInlined())
      return;
    // Function templates are allowed.
    if (FD->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate)
      return;
    // Ignore instantiated functions.
    if (FD->isTemplateInstantiation())
      return;
    // Member function of a class template and member function of a nested class
    // in a class template are allowed.
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
      const auto *DC = MD->getDeclContext();
      while (DC->isRecord()) {
        if (const auto *RD = dyn_cast<CXXRecordDecl>(DC)) {
          if (isa<ClassTemplatePartialSpecializationDecl>(RD))
            return;
          if (RD->getDescribedClassTemplate())
            return;
        }
        DC = DC->getParent();
      }
    }

    bool IsFullSpec = FD->getTemplateSpecializationKind() != TSK_Undeclared;
    diag(FD->getLocation(),
         "%select{function|full function template specialization}0 %1 defined "
         "in a header file; function definitions in header files can lead to "
         "ODR violations")
        << IsFullSpec << FD;
    // inline is not allowed for main function.
    if (FD->isMain())
      return;
    diag(FD->getLocation(), "mark the definition as 'inline'",
         DiagnosticIDs::Note)
        << FixItHint::CreateInsertion(FD->getInnerLocStart(), "inline ");
  } else if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    // C++14 variable templates are allowed.
    if (VD->getDescribedVarTemplate())
      return;
    // Static data members of a class template are allowed.
    if (VD->getDeclContext()->isDependentContext() && VD->isStaticDataMember())
      return;
    // Ignore instantiated static data members of classes.
    if (isTemplateInstantiation(VD->getTemplateSpecializationKind()))
      return;
    // Ignore variable definition within function scope.
    if (VD->hasLocalStorage() || VD->isStaticLocal())
      return;
    // Ignore inline variables.
    if (VD->isInline())
      return;
    // Ignore partial specializations.
    if (isa<VarTemplatePartialSpecializationDecl>(VD))
      return;

    diag(VD->getLocation(),
         "variable %0 defined in a header file; "
         "variable definitions in header files can lead to ODR violations")
        << VD;
  }
}

} // namespace clang::tidy::misc
