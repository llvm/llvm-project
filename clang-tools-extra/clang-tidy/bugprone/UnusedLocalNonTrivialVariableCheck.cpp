//===--- UnusedLocalNonTrivialVariableCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedLocalNonTrivialVariableCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang::tidy::bugprone {

namespace {
static constexpr StringRef DefaultIncludeTypeRegex =
    "::std::.*mutex;::std::future;::std::basic_string;::std::basic_regex;"
    "::std::basic_istringstream;::std::basic_stringstream;::std::bitset;"
    "::std::filesystem::path";

AST_MATCHER(VarDecl, isLocalVarDecl) { return Node.isLocalVarDecl(); }
AST_MATCHER(VarDecl, isReferenced) { return Node.isReferenced(); }
AST_MATCHER(Type, isReferenceType) { return Node.isReferenceType(); }
AST_MATCHER(QualType, isTrivial) {
  return Node.isTrivialType(Finder->getASTContext()) ||
         Node.isTriviallyCopyableType(Finder->getASTContext());
}
} // namespace

UnusedLocalNonTrivialVariableCheck::UnusedLocalNonTrivialVariableCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeTypes(utils::options::parseStringList(
          Options.get("IncludeTypes", DefaultIncludeTypeRegex))),
      ExcludeTypes(
          utils::options::parseStringList(Options.get("ExcludeTypes", ""))) {}

void UnusedLocalNonTrivialVariableCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeTypes",
                utils::options::serializeStringList(IncludeTypes));
  Options.store(Opts, "ExcludeTypes",
                utils::options::serializeStringList(ExcludeTypes));
}

void UnusedLocalNonTrivialVariableCheck::registerMatchers(MatchFinder *Finder) {
  if (IncludeTypes.empty())
    return;

  Finder->addMatcher(
      varDecl(isLocalVarDecl(), unless(isReferenced()),
              unless(isExceptionVariable()), hasLocalStorage(), isDefinition(),
              unless(hasType(isReferenceType())), unless(hasType(isTrivial())),
              unless(hasAttr(attr::Kind::Unused)),
              hasType(hasUnqualifiedDesugaredType(
                  anyOf(recordType(hasDeclaration(namedDecl(
                            matchesAnyListedName(IncludeTypes),
                            unless(matchesAnyListedName(ExcludeTypes))))),
                        templateSpecializationType(hasDeclaration(namedDecl(
                            matchesAnyListedName(IncludeTypes),
                            unless(matchesAnyListedName(ExcludeTypes)))))))))
          .bind("var"),
      this);
}

void UnusedLocalNonTrivialVariableCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("var");
  diag(MatchedDecl->getLocation(), "unused local variable %0 of type %1")
      << MatchedDecl << MatchedDecl->getType();
}

bool UnusedLocalNonTrivialVariableCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind>
UnusedLocalNonTrivialVariableCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

} // namespace clang::tidy::bugprone
