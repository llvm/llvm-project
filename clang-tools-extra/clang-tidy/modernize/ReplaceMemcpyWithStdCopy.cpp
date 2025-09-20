//===--- ReplaceMemcpyWithStdCopy.cpp - clang-tidy----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceMemcpyWithStdCopy.h"
#include "../utils/OptionsUtils.h"
#include <array>

using namespace clang;
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

ReplaceMemcpyWithStdCopy::ReplaceMemcpyWithStdCopy(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM)) {
}

void ReplaceMemcpyWithStdCopy::registerMatchers(MatchFinder *Finder) {
  assert(Finder != nullptr);

  if (!getLangOpts().CPlusPlus)
    return;

  auto MemcpyMatcher =
      callExpr(hasDeclaration(functionDecl(hasName("memcpy"),
                                           isExpansionInSystemHeader())),
               isExpansionInMainFile())
          .bind("memcpy_function");

  Finder->addMatcher(MemcpyMatcher, this);
}

void ReplaceMemcpyWithStdCopy::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  if (!getLangOpts().CPlusPlus)
    return;

  Inserter =
      std::make_unique<utils::IncludeInserter>(SM, getLangOpts(),
                                                       IncludeStyle);
  PP->addPPCallbacks(Inserter->CreatePPCallbacks());
}

void ReplaceMemcpyWithStdCopy::check(const MatchFinder::MatchResult &Result) {
  const auto *MemcpyNode = Result.Nodes.getNodeAs<CallExpr>("memcpy_function");
  assert(MemcpyNode != nullptr);

  DiagnosticBuilder Diag =
      diag(MemcpyNode->getExprLoc(), "use std::copy instead of memcpy");

  renameFunction(Diag, MemcpyNode);
  reorderArgs(Diag, MemcpyNode);
  insertHeader(Diag, MemcpyNode, Result.SourceManager);
}

void ReplaceMemcpyWithStdCopy::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
}

void ReplaceMemcpyWithStdCopy::renameFunction(DiagnosticBuilder &Diag,
                                              const CallExpr *MemcpyNode) {
  const CharSourceRange FunctionNameSourceRange = CharSourceRange::getCharRange(
      MemcpyNode->getBeginLoc(), MemcpyNode->getArg(0)->getBeginLoc());

  Diag << FixItHint::CreateReplacement(FunctionNameSourceRange, "std::copy(");
}

void ReplaceMemcpyWithStdCopy::reorderArgs(DiagnosticBuilder &Diag,
                                           const CallExpr *MemcpyNode) {
  std::array<std::string, 3> arg;

  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  PrintingPolicy Policy(LangOpts);

  // Retrieve all the arguments
  for (uint8_t i = 0; i < arg.size(); i++) {
    llvm::raw_string_ostream s(arg[i]);
    MemcpyNode->getArg(i)->printPretty(s, nullptr, Policy);
  }

  // Create lambda that return SourceRange of an argument
  auto getSourceRange = [MemcpyNode](uint8_t ArgCount) -> SourceRange {
    return SourceRange(MemcpyNode->getArg(ArgCount)->getBeginLoc(),
                       MemcpyNode->getArg(ArgCount)->getEndLoc());
  };

  // Reorder the arguments
  Diag << FixItHint::CreateReplacement(getSourceRange(0), arg[1]);

  arg[2] = arg[1] + " + ((" + arg[2] + ") / sizeof(*(" + arg[1] + ")))";
  Diag << FixItHint::CreateReplacement(getSourceRange(1), arg[2]);

  Diag << FixItHint::CreateReplacement(getSourceRange(2), arg[0]);
}

void ReplaceMemcpyWithStdCopy::insertHeader(DiagnosticBuilder &Diag,
                                            const CallExpr *MemcpyNode,
                                            SourceManager *const SM) {
  Optional<FixItHint> FixInclude = Inserter->CreateIncludeInsertion(
      /*FileID=*/SM->getMainFileID(), /*Header=*/"algorithm",
      /*IsAngled=*/true);
  if (FixInclude)
    Diag << *FixInclude;
}

} // namespace modernize
} // namespace tidy
} // namespace clang
