//===--- UseStdFormatCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdFormatCheck.h"
#include "../utils/FormatStringConverter.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
AST_MATCHER(StringLiteral, isOrdinary) { return Node.isOrdinary(); }
} // namespace

UseStdFormatCheck::UseStdFormatCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)),
      StrFormatLikeFunctions(utils::options::parseStringList(
          Options.get("StrFormatLikeFunctions", ""))),
      ReplacementFormatFunction(
          Options.get("ReplacementFormatFunction", "std::format")),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      MaybeHeaderToInclude(Options.get("FormatHeader")) {
  if (StrFormatLikeFunctions.empty())
    StrFormatLikeFunctions.push_back("absl::StrFormat");

  if (!MaybeHeaderToInclude && ReplacementFormatFunction == "std::format")
    MaybeHeaderToInclude = "<format>";
}

void UseStdFormatCheck::registerPPCallbacks(const SourceManager &SM,
                                            Preprocessor *PP,
                                            Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
  this->PP = PP;
}

void UseStdFormatCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(argumentCountAtLeast(1),
               hasArgument(0, stringLiteral(isOrdinary())),
               callee(functionDecl(matchers::matchesAnyListedName(
                                       StrFormatLikeFunctions))
                          .bind("func_decl")))
          .bind("strformat"),
      this);
}

void UseStdFormatCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  using utils::options::serializeStringList;
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "StrFormatLikeFunctions",
                serializeStringList(StrFormatLikeFunctions));
  Options.store(Opts, "ReplacementFormatFunction", ReplacementFormatFunction);
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  if (MaybeHeaderToInclude)
    Options.store(Opts, "FormatHeader", *MaybeHeaderToInclude);
}

void UseStdFormatCheck::check(const MatchFinder::MatchResult &Result) {
  const unsigned FormatArgOffset = 0;
  const auto *OldFunction = Result.Nodes.getNodeAs<FunctionDecl>("func_decl");
  const auto *StrFormat = Result.Nodes.getNodeAs<CallExpr>("strformat");

  utils::FormatStringConverter::Configuration ConverterConfig;
  ConverterConfig.StrictMode = StrictMode;
  utils::FormatStringConverter Converter(
      Result.Context, StrFormat, FormatArgOffset, ConverterConfig,
      getLangOpts(), *Result.SourceManager, *PP);
  const Expr *StrFormatCall = StrFormat->getCallee();
  if (!Converter.canApply()) {
    diag(StrFormat->getBeginLoc(),
         "unable to use '%0' instead of %1 because %2")
        << StrFormatCall->getSourceRange() << ReplacementFormatFunction
        << OldFunction->getIdentifier()
        << Converter.conversionNotPossibleReason();
    return;
  }

  DiagnosticBuilder Diag =
      diag(StrFormatCall->getBeginLoc(), "use '%0' instead of %1")
      << ReplacementFormatFunction << OldFunction->getIdentifier();
  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(StrFormatCall->getExprLoc(),
                                     StrFormatCall->getEndLoc()),
      ReplacementFormatFunction);
  Converter.applyFixes(Diag, *Result.SourceManager);

  if (MaybeHeaderToInclude)
    Diag << IncludeInserter.createIncludeInsertion(
        Result.Context->getSourceManager().getFileID(
            StrFormatCall->getBeginLoc()),
        *MaybeHeaderToInclude);
}

} // namespace clang::tidy::modernize
