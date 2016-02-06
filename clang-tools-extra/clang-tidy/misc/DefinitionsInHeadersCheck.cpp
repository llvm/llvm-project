//===--- DefinitionsInHeadersCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DefinitionsInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

AST_MATCHER_P(NamedDecl, usesHeaderFileExtension,
              utils::HeaderFileExtensionsSet, HeaderFileExtensions) {
  return utils::isExpansionLocInHeaderFile(
      Node.getLocStart(), Finder->getASTContext().getSourceManager(),
      HeaderFileExtensions);
}

} // namespace

DefinitionsInHeadersCheck::DefinitionsInHeadersCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UseHeaderFileExtension(Options.get("UseHeaderFileExtension", true)),
      RawStringHeaderFileExtensions(
          Options.getLocalOrGlobal("HeaderFileExtensions", ",h,hh,hpp,hxx")) {
  if (!utils::parseHeaderFileExtensions(RawStringHeaderFileExtensions,
                                        HeaderFileExtensions,
                                        ',')) {
    // FIXME: Find a more suitable way to handle invalid configuration
    // options.
    llvm::errs() << "Invalid header file extension: "
                 << RawStringHeaderFileExtensions << "\n";
  }
}

void DefinitionsInHeadersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseHeaderFileExtension", UseHeaderFileExtension);
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void DefinitionsInHeadersCheck::registerMatchers(MatchFinder *Finder) {
  if (UseHeaderFileExtension) {
    Finder->addMatcher(
        namedDecl(anyOf(functionDecl(isDefinition()), varDecl(isDefinition())),
                  usesHeaderFileExtension(HeaderFileExtensions))
            .bind("name-decl"),
        this);
  } else {
    Finder->addMatcher(
        namedDecl(anyOf(functionDecl(isDefinition()), varDecl(isDefinition())),
                  anyOf(usesHeaderFileExtension(HeaderFileExtensions),
                        unless(isExpansionInMainFile())))
            .bind("name-decl"),
        this);
  }
}

void DefinitionsInHeadersCheck::check(const MatchFinder::MatchResult &Result) {
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

  // Internal linkage variable definitions are ignored for now:
  //   const int a = 1;
  //   static int b = 1;
  //
  // Although these might also cause ODR violations, we can be less certain and
  // should try to keep the false-positive rate down.
  if (ND->getLinkageInternal() == InternalLinkage)
    return;

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    // Inline functions are allowed.
    if (FD->isInlined())
      return;
    // Function templates are allowed.
    if (FD->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate)
      return;
    // Function template full specialization is prohibited in header file.
    if (FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
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

    diag(FD->getLocation(),
         "function '%0' defined in a header file; "
         "function definitions in header files can lead to ODR violations")
        << FD->getNameInfo().getName().getAsString()
        << FixItHint::CreateInsertion(FD->getSourceRange().getBegin(),
                                      "inline ");
  } else if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    // Static data members of a class template are allowed.
    if (VD->getDeclContext()->isDependentContext() && VD->isStaticDataMember())
      return;
    if (VD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return;
    // Ignore variable definition within function scope.
    if (VD->hasLocalStorage() || VD->isStaticLocal())
      return;

    diag(VD->getLocation(),
         "variable '%0' defined in a header file; "
         "variable definitions in header files can lead to ODR violations")
        << VD->getName();
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
