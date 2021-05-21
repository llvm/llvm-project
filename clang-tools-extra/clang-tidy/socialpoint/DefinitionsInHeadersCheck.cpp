//===--- DefinitionsInHeadersCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefinitionsInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace socialpoint {

namespace {

AST_MATCHER_P(NamedDecl, usesHeaderFileExtension, utils::FileExtensionsSet,
              HeaderFileExtensions) {
  return utils::isExpansionLocInHeaderFile(
      Node.getBeginLoc(), Finder->getASTContext().getSourceManager(),
      HeaderFileExtensions);
}

llvm::Optional<SourceLocation> findStaticKeywordLocation(const DeclaratorDecl* Decl,
                                                         SourceManager *SM,
                                                         const LangOptions& LangOpts)
{
    llvm::Optional<SourceLocation> loc = Decl->getInnerLocStart();
    while(loc && loc <= Decl->getEndLoc())
    {
        auto Range = CharSourceRange::getTokenRange(*loc);
        if(Lexer::getSourceText(Range, *SM, LangOpts) == "static")
        {
            return loc;
        }
        auto Next = Lexer::findNextToken(*loc, *SM, LangOpts);
        loc = Next.map([](const auto& token) {
            return token.getLocation();
            });
    }
    return llvm::None;
}

} // namespace

DefinitionsInHeadersCheck::DefinitionsInHeadersCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
    IncludeInternalLinkage(Options.get("IncludeInternalLinkage", false)),
    UseHeaderFileExtension(Options.get("UseHeaderFileExtension", true)),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
          "HeaderFileExtensions", utils::defaultHeaderFileExtensions()))
{
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
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
  Options.store(Opts, "IncludeInternalLinkage", IncludeInternalLinkage);
}

void DefinitionsInHeadersCheck::registerMatchers(MatchFinder *Finder) {
  auto DefinitionMatcher =
      anyOf(functionDecl(isDefinition(), unless(isDeleted())),
            varDecl(isDefinition()));
  if (UseHeaderFileExtension) {
    Finder->addMatcher(namedDecl(DefinitionMatcher,
                                 usesHeaderFileExtension(HeaderFileExtensions))
                           .bind("name-decl"),
                       this);
  } else {
    Finder->addMatcher(
        namedDecl(DefinitionMatcher,
                  anyOf(usesHeaderFileExtension(HeaderFileExtensions),
                        unless(isExpansionInMainFile())))
            .bind("name-decl"),
        this);
  }
}

template <typename T>
void DefinitionsInHeadersCheck::Fix(const T* decl,  SourceManager *SM, 
                                    const LangOptions& LangOpts)
{
  if(decl->getStorageClass() == SC_Static)
  {
    if(auto loc = findStaticKeywordLocation(decl, SM, LangOpts)) {
      diag(*loc, /*FixDescription=*/"replace 'static' by 'inline'",
           DiagnosticIDs::Note)
           << FixItHint::CreateReplacement(*loc, "inline");
    } else {
      diag(decl->getInnerLocStart(), "replace 'static' by 'inline'",
           DiagnosticIDs::Note);
    }
  }
  else
  {
    auto TokenCharRange = CharSourceRange::getTokenRange(decl->getInnerLocStart());
    llvm::StringRef Token = Lexer::getSourceText(TokenCharRange, *SM, LangOpts);
    std::string Replacement = ("inline " + Token).str();
    diag(decl->getInnerLocStart(), "make as 'inline'",
         DiagnosticIDs::Note)
         << FixItHint::CreateReplacement(decl->getInnerLocStart(), Replacement);
    }
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

  // Internal linkage variable definitions are ignored if IncludeInternalLinkage
  // is not set:
  //   const int a = 1;
  //   static int b = 1;
  //
  // Although these might also cause ODR violations, we can be less certain and
  // should try to keep the false-positive rate down.
  //
  // FIXME: Should declarations in anonymous namespaces get the same treatment
  // as static / const declarations?
  if (!IncludeInternalLinkage && 
      !ND->hasExternalFormalLinkage() && !ND->isInAnonymousNamespace())
    return;

  SourceManager *SourceManager = Result.SourceManager;
  const LangOptions& LangOpts = Result.Context->getLangOpts();

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    // Inline functions are allowed.
    if (FD->isInlined() || FD->hasAttr<AlwaysInlineAttr>())
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
    Fix(FD, SourceManager, LangOpts);
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
    // Ignore constexpr variables.
    if(VD->isConstexpr())
        return;

    diag(VD->getLocation(),
         "variable %0 defined in a header file; "
         "variable definitions in header files can lead to ODR violations")
        << VD;
    Fix(VD, SourceManager, LangOpts);
  }
}

} // namespace socialpoint
} // namespace tidy
} // namespace clang
