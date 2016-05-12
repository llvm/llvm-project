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
  auto DeclMatcher = hasDeclaration(namedDecl().bind("used"));
  Finder->addMatcher(loc(recordType(DeclMatcher)), this);
  Finder->addMatcher(loc(templateSpecializationType(DeclMatcher)), this);
  Finder->addMatcher(declRefExpr().bind("used"), this);
}

void UnusedUsingDeclsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Using = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    // FIXME: Implement the correct behavior for using declarations with more
    // than one shadow.
    if (Using->shadow_size() != 1)
      return;
    const auto *TargetDecl =
        Using->shadow_begin()->getTargetDecl()->getCanonicalDecl();

    // Ignores using-declarations defined in macros.
    if (TargetDecl->getLocation().isMacroID())
      return;

    // Ignores using-declarations defined in class definition.
    if (isa<CXXRecordDecl>(TargetDecl->getDeclContext()))
      return;

    if (!isa<RecordDecl>(TargetDecl) && !isa<ClassTemplateDecl>(TargetDecl) &&
        !isa<FunctionDecl>(TargetDecl) && !isa<VarDecl>(TargetDecl) &&
        !isa<FunctionTemplateDecl>(TargetDecl))
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
    if (const auto *Specialization =
            dyn_cast<ClassTemplateSpecializationDecl>(Used))
      Used = Specialization->getSpecializedTemplate();
    removeFromFoundDecls(Used);
    return;
  }

  if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("used")) {
    if (const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      if (const auto *FDT = FD->getPrimaryTemplate())
        removeFromFoundDecls(FDT);
      else
        removeFromFoundDecls(FD);
    } else if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      removeFromFoundDecls(VD);
    }
  }
}

void UnusedUsingDeclsCheck::removeFromFoundDecls(const Decl *D) {
  auto I = FoundDecls.find(D->getCanonicalDecl());
  if (I != FoundDecls.end())
    I->second = nullptr;
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
