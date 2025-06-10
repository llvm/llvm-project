//===--- MakeFunctionToDirectCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeFunctionToDirectCheck.h"
#include "../utils/TransformerClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

using namespace ::clang::ast_matchers;
using namespace ::clang::transformer;

namespace clang::tidy::modernize {

namespace {

RewriteRuleWith<std::string> makeFunctionToDirectCheckImpl(
    bool CheckMakePair, bool CheckMakeOptional, bool CheckMakeTuple) {
  std::vector<RewriteRuleWith<std::string>> Rules;

  // Helper to create a rule for a specific make_* function
  auto createRule = [](StringRef MakeFunction, StringRef DirectType) {
    auto WarningMessage = cat("use class template argument deduction (CTAD) "
                              "instead of ", MakeFunction);

    return makeRule(
        callExpr(callee(functionDecl(hasName(MakeFunction))),
                 unless(hasParent(implicitCastExpr(
                     hasImplicitDestinationType(qualType(hasCanonicalType(
                         qualType(asString("void")))))))))
            .bind("make_call"),
        changeTo(node("make_call"), cat(DirectType, "(", callArgs("make_call"), ")")),
        WarningMessage);
  };

  if (CheckMakeOptional) {
    Rules.push_back(createRule("std::make_optional", "std::optional"));
  }

  if (CheckMakePair) {
    Rules.push_back(createRule("std::make_pair", "std::pair"));
  }

  if (CheckMakeTuple) {
    Rules.push_back(createRule("std::make_tuple", "std::tuple"));
  }

  return applyFirst(Rules);
}

} // namespace

MakeFunctionToDirectCheck::MakeFunctionToDirectCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : utils::TransformerClangTidyCheck(Name, Context),
      CheckMakePair(Options.get("CheckMakePair", true)),
      CheckMakeOptional(Options.get("CheckMakeOptional", true)),
      CheckMakeTuple(Options.get("CheckMakeTuple", true)) {
  setRule(makeFunctionToDirectCheckImpl(CheckMakePair, CheckMakeOptional, CheckMakeTuple));
}

void MakeFunctionToDirectCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckMakePair", CheckMakePair);
  Options.store(Opts, "CheckMakeOptional", CheckMakeOptional);
  Options.store(Opts, "CheckMakeTuple", CheckMakeTuple);
}

} // namespace clang::tidy::modernize