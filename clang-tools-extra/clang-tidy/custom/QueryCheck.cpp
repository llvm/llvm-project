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
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::custom {

static void emitConfigurationDiag(ClangTidyContext *Context, StringRef Message,
                                  StringRef CheckName) {
  Context->configurationDiag("%0 in '%1'", DiagnosticIDs::Warning)
      << Message << CheckName;
}

static SmallVector<ast_matchers::dynamic::DynTypedMatcher>
parseQuery(const ClangTidyOptions::CustomCheckValue &V,
           ClangTidyContext *Context) {
  SmallVector<ast_matchers::dynamic::DynTypedMatcher> Matchers{};
  clang::query::QuerySession QS({});
  llvm::StringRef QueryStringRef{V.Query};
  while (!QueryStringRef.empty()) {
    const query::QueryRef Q = query::QueryParser::parse(QueryStringRef, QS);
    switch (Q->Kind) {
    case query::QK_Match: {
      const auto &MatchQuery = llvm::cast<query::MatchQuery>(*Q);
      Matchers.push_back(MatchQuery.Matcher);
      break;
    }
    case query::QK_Let: {
      const auto &LetQuery = llvm::cast<query::LetQuery>(*Q);
      LetQuery.run(llvm::errs(), QS);
      break;
    }
    case query::QK_NoOp: {
      const auto &NoOpQuery = llvm::cast<query::NoOpQuery>(*Q);
      NoOpQuery.run(llvm::errs(), QS);
      break;
    }
    case query::QK_Invalid: {
      const auto &InvalidQuery = llvm::cast<query::InvalidQuery>(*Q);
      emitConfigurationDiag(Context, InvalidQuery.ErrStr, V.Name);
      return {};
    }
    // FIXME: TODO
    case query::QK_File: {
      emitConfigurationDiag(Context, "unsupported query kind 'File'", V.Name);
      return {};
    }
    case query::QK_DisableOutputKind: {
      emitConfigurationDiag(
          Context, "unsupported query kind 'DisableOutputKind'", V.Name);
      return {};
    }
    case query::QK_EnableOutputKind: {
      emitConfigurationDiag(
          Context, "unsupported query kind 'EnableOutputKind'", V.Name);
      return {};
    }
    case query::QK_SetOutputKind: {
      emitConfigurationDiag(Context, "unsupported query kind 'SetOutputKind'",
                            V.Name);
      return {};
    }
    case query::QK_SetTraversalKind: {
      emitConfigurationDiag(
          Context, "unsupported query kind 'SetTraversalKind'", V.Name);
      return {};
    }
    case query::QK_SetBool: {
      emitConfigurationDiag(Context, "unsupported query kind 'SetBool'",
                            V.Name);
      return {};
    }
    case query::QK_Help: {
      emitConfigurationDiag(Context, "unsupported query kind 'Help'", V.Name);
      return {};
    }
    case query::QK_Quit: {
      emitConfigurationDiag(Context, "unsupported query kind 'Quit'", V.Name);
      return {};
    }
    }
    QueryStringRef = Q->RemainingContent;
  }
  return Matchers;
}

QueryCheck::QueryCheck(llvm::StringRef Name,
                       const ClangTidyOptions::CustomCheckValue &V,
                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  for (const ClangTidyOptions::CustomCheckDiag &D : V.Diags) {
    auto DiagnosticIdIt =
        Diags
            .try_emplace(D.Level.value_or(DiagnosticIDs::Warning),
                         llvm::StringMap<llvm::SmallVector<std::string>>{})
            .first;
    auto DiagMessageIt =
        DiagnosticIdIt->getSecond()
            .try_emplace(D.BindName, llvm::SmallVector<std::string>{})
            .first;
    DiagMessageIt->second.emplace_back(D.Message);
  }
  Matchers = parseQuery(V, Context);
}

void QueryCheck::registerMatchers(MatchFinder *Finder) {
  for (const ast_matchers::dynamic::DynTypedMatcher &M : Matchers)
    Finder->addDynamicMatcher(M, this);
}

void QueryCheck::check(const MatchFinder::MatchResult &Result) {
  auto Emit = [this](const DiagMaps &DiagMaps, const std::string &BindName,
                     const DynTypedNode &Node, DiagnosticIDs::Level Level) {
    const DiagMaps::const_iterator DiagMapIt = DiagMaps.find(Level);
    if (DiagMapIt == DiagMaps.end())
      return;
    const BindNameMapToDiagMessage &BindNameMap = DiagMapIt->second;
    const BindNameMapToDiagMessage::const_iterator BindNameMapIt =
        BindNameMap.find(BindName);
    if (BindNameMapIt == BindNameMap.end())
      return;
    for (const std::string &Message : BindNameMapIt->second)
      diag(Node.getSourceRange().getBegin(), Message, Level);
  };
  for (const auto &[Name, Node] : Result.Nodes.getMap())
    Emit(Diags, Name, Node, DiagnosticIDs::Warning);
  // place Note last, otherwise it will not be emitted
  for (const auto &[Name, Node] : Result.Nodes.getMap())
    Emit(Diags, Name, Node, DiagnosticIDs::Note);
}
} // namespace clang::tidy::custom
