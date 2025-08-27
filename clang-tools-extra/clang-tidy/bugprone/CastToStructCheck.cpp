//===--- CastToStructCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CastToStructCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(Type, charType) {
  return Node.isCharType();
}
AST_MATCHER(Type, unionType) {
  return Node.isUnionType();
}

}

CastToStructCheck::CastToStructCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredCasts(
          utils::options::parseStringList(Options.get("IgnoredCasts", ""))) {
  IgnoredCastsRegex.reserve(IgnoredCasts.size());
  for (const auto &Str : IgnoredCasts) {
    std::string WholeWordRegex;
    WholeWordRegex.reserve(Str.size() + 2);
    WholeWordRegex.push_back('^');
    WholeWordRegex.append(Str);
    WholeWordRegex.push_back('$');
    IgnoredCastsRegex.emplace_back(WholeWordRegex);
  }
}

void CastToStructCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredCasts",
                utils::options::serializeStringList(IgnoredCasts));
}

void CastToStructCheck::registerMatchers(MatchFinder *Finder) {
  auto FromPointee =
      qualType(hasUnqualifiedDesugaredType(type().bind("FromType")),
               unless(voidType()),
               unless(charType()),
               unless(unionType()))
          .bind("FromPointee");
  auto ToPointee =
      qualType(hasUnqualifiedDesugaredType(
                   recordType(unless(hasDeclaration(recordDecl(isUnion()))))
                       .bind("ToType")))
          .bind("ToPointee");
  auto FromPtrType = qualType(pointsTo(FromPointee)).bind("FromPtr");
  auto ToPtrType = qualType(pointsTo(ToPointee)).bind("ToPtr");
  Finder->addMatcher(cStyleCastExpr(hasSourceExpression(hasType(FromPtrType)),
                                    hasType(ToPtrType))
                         .bind("CastExpr"),
                     this);
}

void CastToStructCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *const FoundCastExpr =
      Result.Nodes.getNodeAs<CStyleCastExpr>("CastExpr");
  const auto *const FromPtr = Result.Nodes.getNodeAs<QualType>("FromPtr");
  const auto *const ToPtr = Result.Nodes.getNodeAs<QualType>("ToPtr");
  const auto *const FromType = Result.Nodes.getNodeAs<Type>("FromType");
  const auto *const ToType = Result.Nodes.getNodeAs<RecordType>("ToType");

  if (FromType == ToType)
    return;

  auto CheckNameIgnore = [this](const std::string &FromName, const std::string &ToName) {
    bool FromMatch = false;
    for (auto [Idx, Regex] : llvm::enumerate(IgnoredCastsRegex)) {
      if (Idx % 2 == 0) {
        FromMatch = Regex.match(FromName);
      } else {
        if (FromMatch && Regex.match(ToName))
          return true;
      }
    }
    return false;
  };

  if (CheckNameIgnore(FromPtr->getAsString(), ToPtr->getAsString()))
    return;

  diag(FoundCastExpr->getExprLoc(),
       "casting a %0 pointer to a "
       "%1 pointer and accessing a field can lead to memory "
       "access errors or data corruption")
      << *FromPtr << *ToPtr;
}

} // namespace clang::tidy::bugprone
