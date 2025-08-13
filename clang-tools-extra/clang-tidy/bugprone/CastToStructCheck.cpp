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

CastToStructCheck::CastToStructCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredFunctions(
          utils::options::parseStringList(Options.get("IgnoredFunctions", ""))),
      IgnoredFromTypes(
          utils::options::parseStringList(Options.get("IgnoredFromTypes", ""))),
      IgnoredToTypes(
          utils::options::parseStringList(Options.get("IgnoredToTypes", ""))) {}

void CastToStructCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredFunctions",
                utils::options::serializeStringList(IgnoredFunctions));
  Options.store(Opts, "IgnoredFromTypes",
                utils::options::serializeStringList(IgnoredFromTypes));
  Options.store(Opts, "IgnoredToTypes",
                utils::options::serializeStringList(IgnoredToTypes));
}

void CastToStructCheck::registerMatchers(MatchFinder *Finder) {
  auto FromPointee =
      qualType(hasUnqualifiedDesugaredType(type().bind("FromType")),
               unless(qualType(matchers::matchesAnyListedTypeName(
                   IgnoredFromTypes, false))))
          .bind("FromPointee");
  auto ToPointee =
      qualType(hasUnqualifiedDesugaredType(recordType().bind("ToType")),
               unless(qualType(
                   matchers::matchesAnyListedTypeName(IgnoredToTypes, false))))
          .bind("ToPointee");
  auto FromPtrType = qualType(pointsTo(FromPointee)).bind("FromPtr");
  auto ToPtrType = qualType(pointsTo(ToPointee)).bind("ToPtr");
  Finder->addMatcher(
      cStyleCastExpr(hasSourceExpression(hasType(FromPtrType)),
                     hasType(ToPtrType),
                     unless(hasAncestor(functionDecl(
                         matchers::matchesAnyListedName(IgnoredFunctions)))))
          .bind("CastExpr"),
      this);
}

void CastToStructCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *const FoundCastExpr =
      Result.Nodes.getNodeAs<CStyleCastExpr>("CastExpr");
  const auto *const FromPtr = Result.Nodes.getNodeAs<QualType>("FromPtr");
  const auto *const ToPtr = Result.Nodes.getNodeAs<QualType>("ToPtr");
  const auto *const FromPointee =
      Result.Nodes.getNodeAs<QualType>("FromPointee");
  const auto *const ToPointee = Result.Nodes.getNodeAs<QualType>("ToPointee");
  const auto *const FromType = Result.Nodes.getNodeAs<Type>("FromType");
  const auto *const ToType = Result.Nodes.getNodeAs<RecordType>("ToType");
  if (!FromPointee || !ToPointee)
    return;
  if (FromType->isVoidType() || FromType->isUnionType() ||
      ToType->isUnionType())
    return;
  if (FromType == ToType)
    return;
  diag(FoundCastExpr->getExprLoc(),
       "casting a %0 pointer to a "
       "%1 pointer and accessing a field can lead to memory "
       "access errors or data corruption")
      << *FromPtr << *ToPtr;
}

} // namespace clang::tidy::bugprone
