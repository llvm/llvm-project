//===--- QueryCheck.cpp - clang-tidy --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryCheck.h"
#include "../../clang-query/Query.h"
#include "../../clang-query/QueryParser.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::custom {

QueryCheck::QueryCheck(llvm::StringRef Name,
                       const ClangTidyOptions::CustomCheckValue &V,
                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  for (const ClangTidyOptions::CustomCheckDiag &D : V.Diags) {
    auto It = Diags.try_emplace(D.BindName, llvm::SmallVector<Diag>{}).first;
    It->second.emplace_back(
        Diag{D.Message, D.Level.value_or(DiagnosticIDs::Warning)});
  }

  clang::query::QuerySession QS({});
  llvm::StringRef QueryStringRef{V.Query};
  while (!QueryStringRef.empty()) {
    query::QueryRef Q = query::QueryParser::parse(QueryStringRef, QS);
    switch (Q->Kind) {
    case query::QK_Match: {
      const auto &MatchQuerry = llvm::cast<query::MatchQuery>(*Q);
      Matchers.push_back(MatchQuerry.Matcher);
      break;
    }
    case query::QK_Let: {
      const auto &LetQuerry = llvm::cast<query::LetQuery>(*Q);
      LetQuerry.run(llvm::errs(), QS);
      break;
    }
    case query::QK_Invalid: {
      const auto &InvalidQuerry = llvm::cast<query::InvalidQuery>(*Q);
      Context->configurationDiag(InvalidQuerry.ErrStr);
      break;
    }
    // FIXME: TODO
    case query::QK_File:
    case query::QK_DisableOutputKind:
    case query::QK_EnableOutputKind:
    case query::QK_SetOutputKind:
    case query::QK_SetTraversalKind:
    case query::QK_Help:
    case query::QK_NoOp:
    case query::QK_Quit:
    case query::QK_SetBool: {
      Context->configurationDiag("unsupported querry kind");
    }
    }
    QueryStringRef = Q->RemainingContent;
  }
}

void QueryCheck::registerMatchers(MatchFinder *Finder) {
  for (const ast_matchers::dynamic::DynTypedMatcher &M : Matchers)
    Finder->addDynamicMatcher(M, this);
}

void QueryCheck::check(const MatchFinder::MatchResult &Result) {
  for (auto &[Name, Node] : Result.Nodes.getMap())
    if (Diags.contains(Name))
      for (const Diag &D : Diags[Name])
        diag(D.Message, D.Level) << Node.getSourceRange();
}

} // namespace clang::tidy::custom
