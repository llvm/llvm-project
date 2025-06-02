//===--- EmptyCatchCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EmptyCatchCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using ::clang::ast_matchers::internal::Matcher;

namespace clang::tidy::bugprone {

namespace {
AST_MATCHER(CXXCatchStmt, isInMacro) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID() ||
         Node.getCatchLoc().isMacroID();
}

AST_MATCHER_P(CXXCatchStmt, hasHandler, Matcher<Stmt>, InnerMatcher) {
  Stmt *Handler = Node.getHandlerBlock();
  if (!Handler)
    return false;
  return InnerMatcher.matches(*Handler, Finder, Builder);
}

AST_MATCHER_P(CXXCatchStmt, hasCaughtType, Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getCaughtType(), Finder, Builder);
}

AST_MATCHER_P(CompoundStmt, hasAnyTextFromList, std::vector<llvm::StringRef>,
              List) {
  if (List.empty())
    return false;

  ASTContext &Context = Finder->getASTContext();
  SourceManager &SM = Context.getSourceManager();
  StringRef Text = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Node.getSourceRange()), SM,
      Context.getLangOpts());
  return llvm::any_of(List, [&](const StringRef &Str) {
    return Text.contains_insensitive(Str);
  });
}

} // namespace

EmptyCatchCheck::EmptyCatchCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreCatchWithKeywords(utils::options::parseStringList(
          Options.get("IgnoreCatchWithKeywords", "@TODO;@FIXME"))),
      AllowEmptyCatchForExceptions(utils::options::parseStringList(
          Options.get("AllowEmptyCatchForExceptions", ""))) {}

void EmptyCatchCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreCatchWithKeywords",
                utils::options::serializeStringList(IgnoreCatchWithKeywords));
  Options.store(
      Opts, "AllowEmptyCatchForExceptions",
      utils::options::serializeStringList(AllowEmptyCatchForExceptions));
}

bool EmptyCatchCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind> EmptyCatchCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

void EmptyCatchCheck::registerMatchers(MatchFinder *Finder) {
  auto AllowedNamedExceptionDecl =
      namedDecl(matchers::matchesAnyListedName(AllowEmptyCatchForExceptions));
  auto AllowedNamedExceptionTypes =
      qualType(anyOf(hasDeclaration(AllowedNamedExceptionDecl),
                     references(AllowedNamedExceptionDecl),
                     pointsTo(AllowedNamedExceptionDecl)));
  auto IgnoredExceptionType =
      qualType(anyOf(AllowedNamedExceptionTypes,
                     hasCanonicalType(AllowedNamedExceptionTypes)));

  Finder->addMatcher(
      cxxCatchStmt(unless(isExpansionInSystemHeader()), unless(isInMacro()),
                   unless(hasCaughtType(IgnoredExceptionType)),
                   hasHandler(compoundStmt(
                       statementCountIs(0),
                       unless(hasAnyTextFromList(IgnoreCatchWithKeywords)))))
          .bind("catch"),
      this);
}

void EmptyCatchCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCatchStmt = Result.Nodes.getNodeAs<CXXCatchStmt>("catch");

  diag(
      MatchedCatchStmt->getCatchLoc(),
      "empty catch statements hide issues; to handle exceptions appropriately, "
      "consider re-throwing, handling, or avoiding catch altogether");
}

} // namespace clang::tidy::bugprone
