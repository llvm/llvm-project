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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

// FixItHint - Hint to check documentation script to mark this check as
// providing a FixIt.

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

using ::clang::transformer::applyFirst;
using ::clang::transformer::cat;
using ::clang::transformer::change;
using ::clang::transformer::makeRule;
using ::clang::transformer::node;

static const char DefaultStringLikeClasses[] = "::std::basic_string;"
                                               "::std::basic_string_view;"
                                               "::absl::string_view";
static const char DefaultAbseilStringsMatchHeader[] = "absl/strings/match.h";

static llvm::Optional<transformer::RewriteRule>
MakeRule(const LangOptions &LangOpts,
         const ClangTidyCheck::OptionsView &Options) {
  // Parse options.
  //
  // FIXME(tdl-g): These options are being parsed redundantly with the
  // constructor because TransformerClangTidyCheck forces us to provide MakeRule
  // before "this" is fully constructed, but StoreOptions requires us to store
  // the parsed options in "this".  We need to fix TransformerClangTidyCheck and
  // then we can clean this up.
  const std::vector<std::string> StringLikeClassNames =
      utils::options::parseStringList(
          Options.get("StringLikeClasses", DefaultStringLikeClasses));
  const std::string AbseilStringsMatchHeader =
      Options.get("AbseilStringsMatchHeader", DefaultAbseilStringsMatchHeader);

  auto StringLikeClass = cxxRecordDecl(hasAnyName(SmallVector<StringRef, 4>(
      StringLikeClassNames.begin(), StringLikeClassNames.end())));
  auto StringType =
      hasUnqualifiedDesugaredType(recordType(hasDeclaration(StringLikeClass)));
  auto CharStarType =
      hasUnqualifiedDesugaredType(pointerType(pointee(isAnyCharacter())));
  auto StringNpos = declRefExpr(
      to(varDecl(hasName("npos"), hasDeclContext(StringLikeClass))));
  auto StringFind = cxxMemberCallExpr(
      callee(cxxMethodDecl(
          hasName("find"),
          hasParameter(0, parmVarDecl(anyOf(hasType(StringType),
                                            hasType(CharStarType)))))),
      on(hasType(StringType)), hasArgument(0, expr().bind("parameter_to_find")),
      anyOf(hasArgument(1, integerLiteral(equals(0))),
            hasArgument(1, cxxDefaultArgExpr())),
      onImplicitObjectArgument(expr().bind("string_being_searched")));

  tooling::RewriteRule rule = applyFirst(
      {makeRule(binaryOperator(hasOperatorName("=="),
                               hasOperands(ignoringParenImpCasts(StringNpos),
                                           ignoringParenImpCasts(StringFind))),
                change(cat("!absl::StrContains(", node("string_being_searched"),
                           ", ", node("parameter_to_find"), ")")),
                cat("use !absl::StrContains instead of find() == npos")),
       makeRule(binaryOperator(hasOperatorName("!="),
                               hasOperands(ignoringParenImpCasts(StringNpos),
                                           ignoringParenImpCasts(StringFind))),
                change(cat("absl::StrContains(", node("string_being_searched"),
                           ", ", node("parameter_to_find"), ")")),
                cat("use absl::StrContains instead of find() != npos"))});
  addInclude(rule, AbseilStringsMatchHeader);
  return rule;
}

StringFindStrContainsCheck::StringFindStrContainsCheck(
    StringRef Name, ClangTidyContext *Context)
    : TransformerClangTidyCheck(&MakeRule, Name, Context),
      StringLikeClassesOption(utils::options::parseStringList(
          Options.get("StringLikeClasses", DefaultStringLikeClasses))),
      AbseilStringsMatchHeaderOption(Options.get(
          "AbseilStringsMatchHeader", DefaultAbseilStringsMatchHeader)) {}

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

} // namespace abseil
} // namespace tidy
} // namespace clang
