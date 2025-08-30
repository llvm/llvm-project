//===--- StringFindStrContainsCheck.cc - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringFindStrContainsCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

// FixItHint - Hint to check documentation script to mark this check as
// providing a FixIt.

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

using ::clang::transformer::addInclude;
using ::clang::transformer::applyFirst;
using ::clang::transformer::cat;
using ::clang::transformer::changeTo;
using ::clang::transformer::makeRule;
using ::clang::transformer::node;
using ::clang::transformer::RewriteRuleWith;

namespace {
AST_MATCHER(Type, isCharType) { return Node.isCharType(); }
} // namespace

static const char DefaultStringLikeClasses[] = "::std::basic_string;"
                                               "::std::basic_string_view;"
                                               "::absl::string_view";
static const char DefaultAbseilStringsMatchHeader[] = "absl/strings/match.h";

static transformer::RewriteRuleWith<std::string>
makeRewriteRule(ArrayRef<StringRef> StringLikeClassNames,
                StringRef AbseilStringsMatchHeader) {
  auto StringLikeClass = cxxRecordDecl(hasAnyName(StringLikeClassNames));
  auto StringType =
      hasUnqualifiedDesugaredType(recordType(hasDeclaration(StringLikeClass)));
  auto CharStarType =
      hasUnqualifiedDesugaredType(pointerType(pointee(isAnyCharacter())));
  auto CharType = hasUnqualifiedDesugaredType(isCharType());
  auto StringNpos = declRefExpr(
      to(varDecl(hasName("npos"), hasDeclContext(StringLikeClass))));
  auto StringFind = cxxMemberCallExpr(
      callee(cxxMethodDecl(
          hasName("find"), parameterCountIs(2),
          hasParameter(
              0, parmVarDecl(anyOf(hasType(StringType), hasType(CharStarType),
                                   hasType(CharType)))))),
      on(hasType(StringType)), hasArgument(0, expr().bind("parameter_to_find")),
      anyOf(hasArgument(1, integerLiteral(equals(0))),
            hasArgument(1, cxxDefaultArgExpr())),
      onImplicitObjectArgument(expr().bind("string_being_searched")));

  RewriteRuleWith<std::string> Rule = applyFirst(
      {makeRule(
           binaryOperator(hasOperatorName("=="),
                          hasOperands(ignoringParenImpCasts(StringNpos),
                                      ignoringParenImpCasts(StringFind))),
           {changeTo(cat("!absl::StrContains(", node("string_being_searched"),
                         ", ", node("parameter_to_find"), ")")),
            addInclude(AbseilStringsMatchHeader)},
           cat("use !absl::StrContains instead of find() == npos")),
       makeRule(
           binaryOperator(hasOperatorName("!="),
                          hasOperands(ignoringParenImpCasts(StringNpos),
                                      ignoringParenImpCasts(StringFind))),
           {changeTo(cat("absl::StrContains(", node("string_being_searched"),
                         ", ", node("parameter_to_find"), ")")),
            addInclude(AbseilStringsMatchHeader)},
           cat("use absl::StrContains instead "
               "of find() != npos"))});
  return Rule;
}

StringFindStrContainsCheck::StringFindStrContainsCheck(
    StringRef Name, ClangTidyContext *Context)
    : TransformerClangTidyCheck(Name, Context),
      StringLikeClassesOption(utils::options::parseStringList(
          Options.get("StringLikeClasses", DefaultStringLikeClasses))),
      AbseilStringsMatchHeaderOption(Options.get(
          "AbseilStringsMatchHeader", DefaultAbseilStringsMatchHeader)) {
  setRule(
      makeRewriteRule(StringLikeClassesOption, AbseilStringsMatchHeaderOption));
}

bool StringFindStrContainsCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11;
}

void StringFindStrContainsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  TransformerClangTidyCheck::storeOptions(Opts);
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClassesOption));
  Options.store(Opts, "AbseilStringsMatchHeader",
                AbseilStringsMatchHeaderOption);
}

} // namespace clang::tidy::abseil
