//===--- UnsafeFunctionsCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeFunctionsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include <cassert>

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang::tidy::bugprone {

static constexpr llvm::StringLiteral OptionNameReportMoreUnsafeFunctions =
    "ReportMoreUnsafeFunctions";

static constexpr llvm::StringLiteral FunctionNamesWithAnnexKReplacementId =
    "FunctionNamesWithAnnexKReplacement";
static constexpr llvm::StringLiteral FunctionNamesId = "FunctionsNames";
static constexpr llvm::StringLiteral AdditionalFunctionNamesId =
    "AdditionalFunctionsNames";
static constexpr llvm::StringLiteral DeclRefId = "DRE";

static std::optional<std::string>
getAnnexKReplacementFor(StringRef FunctionName) {
  return StringSwitch<std::string>(FunctionName)
      .Case("strlen", "strnlen_s")
      .Case("wcslen", "wcsnlen_s")
      .Default((Twine{FunctionName} + "_s").str());
}

static StringRef getReplacementFor(StringRef FunctionName,
                                   bool IsAnnexKAvailable) {
  if (IsAnnexKAvailable) {
    // Try to find a better replacement from Annex K first.
    StringRef AnnexKReplacementFunction =
        StringSwitch<StringRef>(FunctionName)
            .Cases("asctime", "asctime_r", "asctime_s")
            .Case("gets", "gets_s")
            .Default({});
    if (!AnnexKReplacementFunction.empty())
      return AnnexKReplacementFunction;
  }

  // FIXME: Some of these functions are available in C++ under "std::", and
  // should be matched and suggested.
  return StringSwitch<StringRef>(FunctionName)
      .Cases("asctime", "asctime_r", "strftime")
      .Case("gets", "fgets")
      .Case("rewind", "fseek")
      .Case("setbuf", "setvbuf");
}

static StringRef getReplacementForAdditional(StringRef FunctionName,
                                             bool IsAnnexKAvailable) {
  if (IsAnnexKAvailable) {
    // Try to find a better replacement from Annex K first.
    StringRef AnnexKReplacementFunction = StringSwitch<StringRef>(FunctionName)
                                              .Case("bcopy", "memcpy_s")
                                              .Case("bzero", "memset_s")
                                              .Default({});

    if (!AnnexKReplacementFunction.empty())
      return AnnexKReplacementFunction;
  }

  return StringSwitch<StringRef>(FunctionName)
      .Case("bcmp", "memcmp")
      .Case("bcopy", "memcpy")
      .Case("bzero", "memset")
      .Case("getpw", "getpwuid")
      .Case("vfork", "posix_spawn");
}

/// \returns The rationale for replacing the function \p FunctionName with the
/// safer alternative.
static StringRef getRationaleFor(StringRef FunctionName) {
  return StringSwitch<StringRef>(FunctionName)
      .Cases("asctime", "asctime_r", "ctime",
             "is not bounds-checking and non-reentrant")
      .Cases("bcmp", "bcopy", "bzero", "is deprecated")
      .Cases("fopen", "freopen", "has no exclusive access to the opened file")
      .Case("gets", "is insecure, was deprecated and removed in C11 and C++14")
      .Case("getpw", "is dangerous as it may overflow the provided buffer")
      .Cases("rewind", "setbuf", "has no error detection")
      .Case("vfork", "is insecure as it can lead to denial of service "
                     "situations in the parent process")
      .Default("is not bounds-checking");
}

/// Calculates whether Annex K is available for the current translation unit
/// based on the macro definitions and the language options.
///
/// The result is cached and saved in \p CacheVar.
static bool isAnnexKAvailable(std::optional<bool> &CacheVar, Preprocessor *PP,
                              const LangOptions &LO) {
  if (CacheVar.has_value())
    return *CacheVar;

  if (!LO.C11)
    // TODO: How is "Annex K" available in C++ mode?
    return (CacheVar = false).value();

  assert(PP && "No Preprocessor registered.");

  if (!PP->isMacroDefined("__STDC_LIB_EXT1__") ||
      !PP->isMacroDefined("__STDC_WANT_LIB_EXT1__"))
    return (CacheVar = false).value();

  const auto *MI =
      PP->getMacroInfo(PP->getIdentifierInfo("__STDC_WANT_LIB_EXT1__"));
  if (!MI || MI->tokens_empty())
    return (CacheVar = false).value();

  const Token &T = MI->tokens().back();
  if (!T.isLiteral() || !T.getLiteralData())
    return (CacheVar = false).value();

  CacheVar = StringRef(T.getLiteralData(), T.getLength()) == "1";
  return CacheVar.value();
}

UnsafeFunctionsCheck::UnsafeFunctionsCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ReportMoreUnsafeFunctions(
          Options.get(OptionNameReportMoreUnsafeFunctions, true)) {}

void UnsafeFunctionsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, OptionNameReportMoreUnsafeFunctions,
                ReportMoreUnsafeFunctions);
}

void UnsafeFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  if (getLangOpts().C11) {
    // Matching functions with safe replacements only in Annex K.
    auto FunctionNamesWithAnnexKReplacementMatcher = hasAnyName(
        "::bsearch", "::ctime", "::fopen", "::fprintf", "::freopen", "::fscanf",
        "::fwprintf", "::fwscanf", "::getenv", "::gmtime", "::localtime",
        "::mbsrtowcs", "::mbstowcs", "::memcpy", "::memmove", "::memset",
        "::printf", "::qsort", "::scanf", "::snprintf", "::sprintf", "::sscanf",
        "::strcat", "::strcpy", "::strerror", "::strlen", "::strncat",
        "::strncpy", "::strtok", "::swprintf", "::swscanf", "::vfprintf",
        "::vfscanf", "::vfwprintf", "::vfwscanf", "::vprintf", "::vscanf",
        "::vsnprintf", "::vsprintf", "::vsscanf", "::vswprintf", "::vswscanf",
        "::vwprintf", "::vwscanf", "::wcrtomb", "::wcscat", "::wcscpy",
        "::wcslen", "::wcsncat", "::wcsncpy", "::wcsrtombs", "::wcstok",
        "::wcstombs", "::wctomb", "::wmemcpy", "::wmemmove", "::wprintf",
        "::wscanf");
    Finder->addMatcher(
        declRefExpr(to(functionDecl(FunctionNamesWithAnnexKReplacementMatcher)
                           .bind(FunctionNamesWithAnnexKReplacementId)))
            .bind(DeclRefId),
        this);
  }

  // Matching functions with replacements without Annex K.
  auto FunctionNamesMatcher =
      hasAnyName("::asctime", "asctime_r", "::gets", "::rewind", "::setbuf");
  Finder->addMatcher(
      declRefExpr(to(functionDecl(FunctionNamesMatcher).bind(FunctionNamesId)))
          .bind(DeclRefId),
      this);

  if (ReportMoreUnsafeFunctions) {
    // Matching functions with replacements without Annex K, at user request.
    auto AdditionalFunctionNamesMatcher =
        hasAnyName("::bcmp", "::bcopy", "::bzero", "::getpw", "::vfork");
    Finder->addMatcher(
        declRefExpr(to(functionDecl(AdditionalFunctionNamesMatcher)
                           .bind(AdditionalFunctionNamesId)))
            .bind(DeclRefId),
        this);
  }
}

void UnsafeFunctionsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>(DeclRefId);
  const auto *FuncDecl = cast<FunctionDecl>(DeclRef->getDecl());
  assert(DeclRef && FuncDecl && "No valid matched node in check()");

  const auto *AnnexK = Result.Nodes.getNodeAs<FunctionDecl>(
      FunctionNamesWithAnnexKReplacementId);
  const auto *Normal = Result.Nodes.getNodeAs<FunctionDecl>(FunctionNamesId);
  const auto *Additional =
      Result.Nodes.getNodeAs<FunctionDecl>(AdditionalFunctionNamesId);
  assert((AnnexK || Normal || Additional) && "No valid match category.");

  bool AnnexKIsAvailable =
      isAnnexKAvailable(IsAnnexKAvailable, PP, getLangOpts());
  StringRef FunctionName = FuncDecl->getName();
  const std::optional<std::string> ReplacementFunctionName =
      [&]() -> std::optional<std::string> {
    if (AnnexK) {
      if (AnnexKIsAvailable)
        return getAnnexKReplacementFor(FunctionName);
      return std::nullopt;
    }

    if (Normal)
      return getReplacementFor(FunctionName, AnnexKIsAvailable).str();

    if (Additional)
      return getReplacementForAdditional(FunctionName, AnnexKIsAvailable).str();

    llvm_unreachable("Unhandled match category");
  }();
  if (!ReplacementFunctionName)
    return;

  diag(DeclRef->getExprLoc(), "function %0 %1; '%2' should be used instead")
      << FuncDecl << getRationaleFor(FunctionName)
      << ReplacementFunctionName.value() << DeclRef->getSourceRange();
}

void UnsafeFunctionsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP,
    Preprocessor * /*ModuleExpanderPP*/) {
  this->PP = PP;
}

void UnsafeFunctionsCheck::onEndOfTranslationUnit() {
  this->PP = nullptr;
  IsAnnexKAvailable.reset();
}

} // namespace clang::tidy::bugprone
