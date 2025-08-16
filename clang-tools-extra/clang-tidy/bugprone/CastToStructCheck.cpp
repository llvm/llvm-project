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
      IgnoredCasts(
          utils::options::parseStringList(Options.get("IgnoredCasts", ""))) {}

void CastToStructCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredCasts",
                utils::options::serializeStringList(IgnoredCasts));
}

void CastToStructCheck::registerMatchers(MatchFinder *Finder) {
  auto FromPointee =
      qualType(hasUnqualifiedDesugaredType(type().bind("FromType")),
               unless(voidType()),
               unless(hasDeclaration(recordDecl(isUnion()))))
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
  const auto *const FromPointee =
      Result.Nodes.getNodeAs<QualType>("FromPointee");
  const auto *const ToPointee = Result.Nodes.getNodeAs<QualType>("ToPointee");
  const auto *const FromType = Result.Nodes.getNodeAs<Type>("FromType");
  const auto *const ToType = Result.Nodes.getNodeAs<RecordType>("ToType");

  if (FromType == ToType)
    return;

  const std::string FromName = FromPointee->getAsString();
  const std::string ToName = ToPointee->getAsString();
  llvm::Regex FromR;
  llvm::Regex ToR;
  for (auto [Idx, Str] : llvm::enumerate(IgnoredCasts)) {
    if (Idx % 2 == 0) {
      FromR = llvm::Regex(Str);
    } else {
      ToR = llvm::Regex(Str);
      if (FromR.match(FromName) && ToR.match(ToName))
        return;
    }
  }

  diag(FoundCastExpr->getExprLoc(),
       "casting a %0 pointer to a "
       "%1 pointer and accessing a field can lead to memory "
       "access errors or data corruption")
      << *FromPtr << *ToPtr;
}

} // namespace clang::tidy::bugprone
