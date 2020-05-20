//===--- DeprecatedIosBaseAliasesCheck.cpp - clang-tidy--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeprecatedIosBaseAliasesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static constexpr std::array<StringRef, 5> DeprecatedTypes = {
    "::std::ios_base::io_state", "::std::ios_base::open_mode",
    "::std::ios_base::seek_dir", "::std::ios_base::streamoff",
    "::std::ios_base::streampos"};

static llvm::Optional<const char *> getReplacementType(StringRef Type) {
  return llvm::StringSwitch<llvm::Optional<const char *>>(Type)
      .Case("io_state", "iostate")
      .Case("open_mode", "openmode")
      .Case("seek_dir", "seekdir")
      .Default(llvm::None);
}

void DeprecatedIosBaseAliasesCheck::registerMatchers(MatchFinder *Finder) {
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
  auto Replacement = getReplacementType(TypeName);

  const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("TypeLoc");
  SourceLocation IoStateLoc = TL->getBeginLoc();

  // Do not generate fixits for matches depending on template arguments and
  // macro expansions.
  bool Fix = Replacement && !TL->getType()->isDependentType();
  if (IoStateLoc.isMacroID()) {
    IoStateLoc = SM.getSpellingLoc(IoStateLoc);
    Fix = false;
  }

  SourceLocation EndLoc = IoStateLoc.getLocWithOffset(TypeName.size() - 1);

  if (Replacement) {
    auto FixName = *Replacement;
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
