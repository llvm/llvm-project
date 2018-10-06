//===--- DeprecatedIosBaseAliasesCheck.cpp - clang-tidy--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DeprecatedIosBaseAliasesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static const std::array<StringRef, 5> DeprecatedTypes = {
    "::std::ios_base::io_state",  "::std::ios_base::open_mode",
    "::std::ios_base::seek_dir",  "::std::ios_base::streamoff",
    "::std::ios_base::streampos",
};

static const llvm::StringMap<StringRef> ReplacementTypes = {
    {"io_state", "iostate"},
    {"open_mode", "openmode"},
    {"seek_dir", "seekdir"}};

void DeprecatedIosBaseAliasesCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  auto IoStateDecl = typedefDecl(hasAnyName(DeprecatedTypes)).bind("TypeDecl");
  auto IoStateType =
      qualType(hasDeclaration(IoStateDecl), unless(elaboratedType()));

  Finder->addMatcher(typeLoc(loc(IoStateType)).bind("TypeLoc"), this);
}

void DeprecatedIosBaseAliasesCheck::check(
    const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;

  const auto *Typedef = Result.Nodes.getNodeAs<TypedefDecl>("TypeDecl");
  StringRef TypeName = Typedef->getName();
  bool HasReplacement = ReplacementTypes.count(TypeName);

  const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("TypeLoc");
  SourceLocation IoStateLoc = TL->getBeginLoc();

  // Do not generate fixits for matches depending on template arguments and
  // macro expansions.
  bool Fix = HasReplacement && !TL->getType()->isDependentType();
  if (IoStateLoc.isMacroID()) {
    IoStateLoc = SM.getSpellingLoc(IoStateLoc);
    Fix = false;
  }

  SourceLocation EndLoc = IoStateLoc.getLocWithOffset(TypeName.size() - 1);

  if (HasReplacement) {
    auto FixName = ReplacementTypes.lookup(TypeName);
    auto Builder = diag(IoStateLoc, "'std::ios_base::%0' is deprecated; use "
                                    "'std::ios_base::%1' instead")
                   << TypeName << FixName;

    if (Fix)
      Builder << FixItHint::CreateReplacement(SourceRange(IoStateLoc, EndLoc),
                                              FixName);
  } else
    diag(IoStateLoc, "'std::ios_base::%0' is deprecated") << TypeName;
}

} // namespace modernize
} // namespace tidy
} // namespace clang
