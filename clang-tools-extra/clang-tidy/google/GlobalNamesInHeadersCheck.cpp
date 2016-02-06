//===--- GlobalNamesInHeadersCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GlobalNamesInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace readability {

GlobalNamesInHeadersCheck::GlobalNamesInHeadersCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(
          Options.getLocalOrGlobal("HeaderFileExtensions", "h")) {
  if (!utils::parseHeaderFileExtensions(RawStringHeaderFileExtensions,
                                        HeaderFileExtensions,
                                        ',')) {
    llvm::errs() << "Invalid header file extension: "
                 << RawStringHeaderFileExtensions << "\n";
  }
}

void GlobalNamesInHeadersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void
GlobalNamesInHeadersCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      decl(anyOf(usingDecl(), usingDirectiveDecl()),
           hasDeclContext(translationUnitDecl())).bind("using_decl"),
      this);
}

void GlobalNamesInHeadersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<Decl>("using_decl");
  // If it comes from a macro, we'll assume it is fine.
  if (D->getLocStart().isMacroID())
    return;

  // Ignore if it comes from the "main" file ...
  if (Result.SourceManager->isInMainFile(
          Result.SourceManager->getExpansionLoc(D->getLocStart()))) {
    // unless that file is a header.
    if (!utils::isSpellingLocInHeaderFile(
            D->getLocStart(), *Result.SourceManager, HeaderFileExtensions))
      return;
  }

  if (const auto* UsingDirective = dyn_cast<UsingDirectiveDecl>(D)) {
    if (UsingDirective->getNominatedNamespace()->isAnonymousNamespace()) {
      // Anynoumous namespaces inject a using directive into the AST to import
      // the names into the containing namespace.
      // We should not have them in headers, but there is another warning for
      // that.
      return;
    }
  }

  diag(D->getLocStart(),
       "using declarations in the global namespace in headers are prohibited");
}

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang
