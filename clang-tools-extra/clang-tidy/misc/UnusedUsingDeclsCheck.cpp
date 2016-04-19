//===--- UnusedUsingDeclsCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnusedUsingDeclsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void UnusedUsingDeclsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(usingDecl(isExpansionInMainFile()).bind("using"), this);
  Finder->addMatcher(recordType(hasDeclaration(namedDecl().bind("used"))),
                     this);
}

void UnusedUsingDeclsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Using = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    // FIXME: Implement the correct behavior for using declarations with more
    // than one shadow.
    if (Using->shadow_size() != 1)
      return;
    const auto* TargetDecl = Using->shadow_begin()->getTargetDecl();

    // FIXME: Handle other target types.
    if (!isa<RecordDecl>(TargetDecl))
      return;

    FoundDecls[TargetDecl] = Using;
    FoundRanges[TargetDecl] = CharSourceRange::getCharRange(
        Using->getLocStart(),
        Lexer::findLocationAfterToken(
            Using->getLocEnd(), tok::semi, *Result.SourceManager,
            Result.Context->getLangOpts(),
            /*SkipTrailingWhitespaceAndNewLine=*/true));
    return;
  }

  // Mark using declarations as used by setting FoundDecls' value to zero. As
  // the AST is walked in order, usages are only marked after a the
  // corresponding using declaration has been found.
  // FIXME: This currently doesn't look at whether the type reference is
  // actually found with the help of the using declaration.
  if (const auto *Used = Result.Nodes.getNodeAs<NamedDecl>("used")) {
    if (FoundDecls.find(Used) != FoundDecls.end())
      FoundDecls[Used] = nullptr;
  }
}

void UnusedUsingDeclsCheck::onEndOfTranslationUnit() {
  for (const auto &FoundDecl : FoundDecls) {
    if (FoundDecl.second == nullptr)
      continue;
    diag(FoundDecl.second->getLocation(), "using decl %0 is unused")
        << FoundDecl.second
        << FixItHint::CreateRemoval(FoundRanges[FoundDecl.first]);
  }
  FoundDecls.clear();
}

} // namespace misc
} // namespace tidy
} // namespace clang
