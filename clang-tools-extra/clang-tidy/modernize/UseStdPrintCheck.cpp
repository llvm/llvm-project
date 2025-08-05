//===--- UseStdPrintCheck.cpp - clang-tidy-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdPrintCheck.h"
#include "../utils/FormatStringConverter.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
AST_MATCHER(StringLiteral, isOrdinary) { return Node.isOrdinary(); }
} // namespace

UseStdPrintCheck::UseStdPrintCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), PP(nullptr),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)),
      PrintfLikeFunctions(utils::options::parseStringList(
          Options.get("PrintfLikeFunctions", ""))),
      FprintfLikeFunctions(utils::options::parseStringList(
          Options.get("FprintfLikeFunctions", ""))),
      ReplacementPrintFunction(
          Options.get("ReplacementPrintFunction", "std::print")),
      ReplacementPrintlnFunction(
          Options.get("ReplacementPrintlnFunction", "std::println")),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      MaybeHeaderToInclude(Options.get("PrintHeader")) {

  if (PrintfLikeFunctions.empty() && FprintfLikeFunctions.empty()) {
    PrintfLikeFunctions.emplace_back("::printf");
    PrintfLikeFunctions.emplace_back("absl::PrintF");
    FprintfLikeFunctions.emplace_back("::fprintf");
    FprintfLikeFunctions.emplace_back("absl::FPrintF");
  }

  if (!MaybeHeaderToInclude && (ReplacementPrintFunction == "std::print" ||
                                ReplacementPrintlnFunction == "std::println"))
    MaybeHeaderToInclude = "<print>";
}

void UseStdPrintCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  using utils::options::serializeStringList;
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "PrintfLikeFunctions",
                serializeStringList(PrintfLikeFunctions));
  Options.store(Opts, "FprintfLikeFunctions",
                serializeStringList(FprintfLikeFunctions));
  Options.store(Opts, "ReplacementPrintFunction", ReplacementPrintFunction);
  Options.store(Opts, "ReplacementPrintlnFunction", ReplacementPrintlnFunction);
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  if (MaybeHeaderToInclude)
    Options.store(Opts, "PrintHeader", *MaybeHeaderToInclude);
}

void UseStdPrintCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
  this->PP = PP;
}

static clang::ast_matchers::StatementMatcher
unusedReturnValue(clang::ast_matchers::StatementMatcher MatchedCallExpr) {
  auto UnusedInCompoundStmt =
      compoundStmt(forEach(MatchedCallExpr),
                   // The checker can't currently differentiate between the
                   // return statement and other statements inside GNU statement
                   // expressions, so disable the checker inside them to avoid
                   // false positives.
                   unless(hasParent(stmtExpr())));
  auto UnusedInIfStmt =
      ifStmt(eachOf(hasThen(MatchedCallExpr), hasElse(MatchedCallExpr)));
  auto UnusedInWhileStmt = whileStmt(hasBody(MatchedCallExpr));
  auto UnusedInDoStmt = doStmt(hasBody(MatchedCallExpr));
  auto UnusedInForStmt =
      forStmt(eachOf(hasLoopInit(MatchedCallExpr),
                     hasIncrement(MatchedCallExpr), hasBody(MatchedCallExpr)));
  auto UnusedInRangeForStmt = cxxForRangeStmt(hasBody(MatchedCallExpr));
  auto UnusedInCaseStmt = switchCase(forEach(MatchedCallExpr));

  return stmt(anyOf(UnusedInCompoundStmt, UnusedInIfStmt, UnusedInWhileStmt,
                    UnusedInDoStmt, UnusedInForStmt, UnusedInRangeForStmt,
                    UnusedInCaseStmt));
}

void UseStdPrintCheck::registerMatchers(MatchFinder *Finder) {
  if (!PrintfLikeFunctions.empty())
    Finder->addMatcher(
        unusedReturnValue(
            callExpr(argumentCountAtLeast(1),
                     hasArgument(0, stringLiteral(isOrdinary())),
                     callee(functionDecl(matchers::matchesAnyListedName(
                                             PrintfLikeFunctions))
                                .bind("func_decl")))
                .bind("printf")),
        this);

  if (!FprintfLikeFunctions.empty())
    Finder->addMatcher(
        unusedReturnValue(
            callExpr(argumentCountAtLeast(2),
                     hasArgument(1, stringLiteral(isOrdinary())),
                     callee(functionDecl(matchers::matchesAnyListedName(
                                             FprintfLikeFunctions))
                                .bind("func_decl")))
                .bind("fprintf")),
        this);
}

void UseStdPrintCheck::check(const MatchFinder::MatchResult &Result) {
  unsigned FormatArgOffset = 0;
  const auto *OldFunction = Result.Nodes.getNodeAs<FunctionDecl>("func_decl");
  const auto *Printf = Result.Nodes.getNodeAs<CallExpr>("printf");
  if (!Printf) {
    Printf = Result.Nodes.getNodeAs<CallExpr>("fprintf");
    FormatArgOffset = 1;
  }

  utils::FormatStringConverter::Configuration ConverterConfig;
  ConverterConfig.StrictMode = StrictMode;
  ConverterConfig.AllowTrailingNewlineRemoval = true;
  assert(PP && "Preprocessor should be set by registerPPCallbacks");
  utils::FormatStringConverter Converter(
      Result.Context, Printf, FormatArgOffset, ConverterConfig, getLangOpts(),
      *Result.SourceManager, *PP);
  const Expr *PrintfCall = Printf->getCallee();
  const StringRef ReplacementFunction = Converter.usePrintNewlineFunction()
                                            ? ReplacementPrintlnFunction
                                            : ReplacementPrintFunction;
  if (!Converter.canApply()) {
    diag(PrintfCall->getBeginLoc(),
         "unable to use '%0' instead of %1 because %2")
        << PrintfCall->getSourceRange() << ReplacementFunction
        << OldFunction->getIdentifier()
        << Converter.conversionNotPossibleReason();
    return;
  }

  DiagnosticBuilder Diag =
      diag(PrintfCall->getBeginLoc(), "use '%0' instead of %1")
      << ReplacementFunction << OldFunction->getIdentifier();

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(PrintfCall->getExprLoc(),
                                     PrintfCall->getEndLoc()),
      ReplacementFunction);
  Converter.applyFixes(Diag, *Result.SourceManager);

  if (MaybeHeaderToInclude)
    Diag << IncludeInserter.createIncludeInsertion(
        Result.Context->getSourceManager().getFileID(PrintfCall->getBeginLoc()),
        *MaybeHeaderToInclude);
}

} // namespace clang::tidy::modernize
