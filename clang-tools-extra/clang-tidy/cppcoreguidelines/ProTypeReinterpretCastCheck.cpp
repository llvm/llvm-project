//===--- ProTypeReinterpretCastCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeReinterpretCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

static bool isCastToBytes(ASTContext const &Ctx,
                          CXXReinterpretCastExpr const &Expr) {
  // https://eel.is/c++draft/basic.lval#11.3
  static constexpr std::array<StringRef, 3> AllowedByteTypes = {
      "char",
      "unsigned char",
      "std::byte",
  };

  // We only care about pointer casts
  QualType DestType = Expr.getTypeAsWritten();
  if (!DestType->isPointerType())
    return false;

  // Get the unqualified canonical type, and check if it's allowed
  // We need to wrap the Type into a QualType to call getAsString()
  const Type *UnqualDestType =
      DestType.getCanonicalType()->getPointeeType().getTypePtr();
  std::string DestTypeString = QualType(UnqualDestType, /*Quals=*/0)
                                   .getAsString(Ctx.getPrintingPolicy());
  return llvm::any_of(AllowedByteTypes, [DestTypeString](StringRef Type) {
    return Type == DestTypeString;
  });
}

ProTypeReinterpretCastCheck::ProTypeReinterpretCastCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowCastToBytes(Options.getLocalOrGlobal("AllowCastToBytes", false)) {}

void ProTypeReinterpretCastCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowCastToBytes", AllowCastToBytes);
}

void ProTypeReinterpretCastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxReinterpretCastExpr().bind("cast"), this);
}

void ProTypeReinterpretCastCheck::check(
    const MatchFinder::MatchResult &Result) {

  if (const auto *MatchedCast =
          Result.Nodes.getNodeAs<CXXReinterpretCastExpr>("cast")) {
    ASTContext const &Ctx = *Result.Context;
    if (AllowCastToBytes && isCastToBytes(Ctx, *MatchedCast))
      return;

    diag(MatchedCast->getOperatorLoc(), "do not use reinterpret_cast");
  }
}

} // namespace clang::tidy::cppcoreguidelines
