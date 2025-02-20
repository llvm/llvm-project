//===--- UnsafeFunctionsCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeFunctionsCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include <cassert>

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang::tidy::bugprone {

static constexpr llvm::StringLiteral OptionNameCustomFunctions =
    "CustomFunctions";
static constexpr llvm::StringLiteral OptionNameReportDefaultFunctions =
    "ReportDefaultFunctions";
static constexpr llvm::StringLiteral OptionNameReportMoreUnsafeFunctions =
    "ReportMoreUnsafeFunctions";

static constexpr llvm::StringLiteral FunctionNamesWithAnnexKReplacementId =
    "FunctionNamesWithAnnexKReplacement";
static constexpr llvm::StringLiteral FunctionNamesId = "FunctionsNames";
static constexpr llvm::StringLiteral AdditionalFunctionNamesId =
    "AdditionalFunctionsNames";
static constexpr llvm::StringLiteral CustomFunctionNamesId =
    "CustomFunctionNames";
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

static std::vector<UnsafeFunctionsCheck::CheckedFunction>
parseCheckedFunctions(StringRef Option, ClangTidyContext *Context) {
  const std::vector<StringRef> Functions =
      utils::options::parseStringList(Option);
  std::vector<UnsafeFunctionsCheck::CheckedFunction> Result;
  Result.reserve(Functions.size());

  for (StringRef Function : Functions) {
    if (Function.empty())
      continue;

    const auto [Name, Rest] = Function.split(',');
    const auto [Replacement, Reason] = Rest.split(',');

    if (Name.trim().empty()) {
      Context->configurationDiag("invalid configuration value for option '%0'; "
                                 "expected the name of an unsafe function")
          << OptionNameCustomFunctions;
      continue;
    }

    Result.push_back(
        {Name.trim().str(),
         matchers::MatchesAnyListedNameMatcher::NameMatcher(Name.trim()),
         Replacement.trim().str(), Reason.trim().str()});
  }

  return Result;
}

static std::string serializeCheckedFunctions(
    const std::vector<UnsafeFunctionsCheck::CheckedFunction> &Functions) {
  std::vector<std::string> Result;
  Result.reserve(Functions.size());

  for (const auto &Entry : Functions) {
    if (Entry.Reason.empty())
      Result.push_back(Entry.Name + "," + Entry.Replacement);
    else
      Result.push_back(Entry.Name + "," + Entry.Replacement + "," +
                       Entry.Reason);
  }

  return llvm::join(Result, ";");
}

UnsafeFunctionsCheck::UnsafeFunctionsCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CustomFunctions(parseCheckedFunctions(
          Options.get(OptionNameCustomFunctions, ""), Context)),
      ReportDefaultFunctions(
          Options.get(OptionNameReportDefaultFunctions, true)),
      ReportMoreUnsafeFunctions(
          Options.get(OptionNameReportMoreUnsafeFunctions, true)) {}

void UnsafeFunctionsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, OptionNameCustomFunctions,
                serializeCheckedFunctions(CustomFunctions));
  Options.store(Opts, OptionNameReportDefaultFunctions, ReportDefaultFunctions);
  Options.store(Opts, OptionNameReportMoreUnsafeFunctions,
                ReportMoreUnsafeFunctions);
}

void UnsafeFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  if (ReportDefaultFunctions) {
    if (getLangOpts().C11) {
      // Matching functions with safe replacements only in Annex K.
      auto FunctionNamesWithAnnexKReplacementMatcher = hasAnyName(
          "::bsearch", "::ctime", "::fopen", "::fprintf", "::freopen",
          "::fscanf", "::fwprintf", "::fwscanf", "::getenv", "::gmtime",
          "::localtime", "::mbsrtowcs", "::mbstowcs", "::memcpy", "::memmove",
          "::memset", "::printf", "::qsort", "::scanf", "::snprintf",
          "::sprintf", "::sscanf", "::strcat", "::strcpy", "::strerror",
          "::strlen", "::strncat", "::strncpy", "::strtok", "::swprintf",
          "::swscanf", "::vfprintf", "::vfscanf", "::vfwprintf", "::vfwscanf",
          "::vprintf", "::vscanf", "::vsnprintf", "::vsprintf", "::vsscanf",
          "::vswprintf", "::vswscanf", "::vwprintf", "::vwscanf", "::wcrtomb",
          "::wcscat", "::wcscpy", "::wcslen", "::wcsncat", "::wcsncpy",
          "::wcsrtombs", "::wcstok", "::wcstombs", "::wctomb", "::wmemcpy",
          "::wmemmove", "::wprintf", "::wscanf");
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
        declRefExpr(
            to(functionDecl(FunctionNamesMatcher).bind(FunctionNamesId)))
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

  if (!CustomFunctions.empty()) {
    std::vector<llvm::StringRef> FunctionNames;
    FunctionNames.reserve(CustomFunctions.size());

    for (const auto &Entry : CustomFunctions)
      FunctionNames.push_back(Entry.Name);

    auto CustomFunctionsMatcher = matchers::matchesAnyListedName(FunctionNames);

    Finder->addMatcher(declRefExpr(to(functionDecl(CustomFunctionsMatcher)
                                          .bind(CustomFunctionNamesId)))
                           .bind(DeclRefId),
                       this);
    // C++ member calls do not contain a DeclRefExpr to the function decl.
    // Instead, they contain a MemberExpr that refers to the decl.
    Finder->addMatcher(memberExpr(member(functionDecl(CustomFunctionsMatcher)
                                             .bind(CustomFunctionNamesId)))
                           .bind(DeclRefId),
                       this);
  }
}

void UnsafeFunctionsCheck::check(const MatchFinder::MatchResult &Result) {
  const Expr *SourceExpr;
  const FunctionDecl *FuncDecl;

  if (const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>(DeclRefId)) {
    SourceExpr = DeclRef;
    FuncDecl = cast<FunctionDecl>(DeclRef->getDecl());
  } else if (const auto *Member =
                 Result.Nodes.getNodeAs<MemberExpr>(DeclRefId)) {
    SourceExpr = Member;
    FuncDecl = cast<FunctionDecl>(Member->getMemberDecl());
  } else {
    llvm_unreachable("No valid matched node in check()");
    return;
  }

  assert(SourceExpr && FuncDecl && "No valid matched node in check()");

  // Only one of these are matched at a time.
  const auto *AnnexK = Result.Nodes.getNodeAs<FunctionDecl>(
      FunctionNamesWithAnnexKReplacementId);
  const auto *Normal = Result.Nodes.getNodeAs<FunctionDecl>(FunctionNamesId);
  const auto *Additional =
      Result.Nodes.getNodeAs<FunctionDecl>(AdditionalFunctionNamesId);
  const auto *Custom =
      Result.Nodes.getNodeAs<FunctionDecl>(CustomFunctionNamesId);
  assert((AnnexK || Normal || Additional || Custom) &&
         "No valid match category.");

  bool AnnexKIsAvailable =
      isAnnexKAvailable(IsAnnexKAvailable, PP, getLangOpts());
  StringRef FunctionName = FuncDecl->getName();

  if (Custom) {
    for (const auto &Entry : CustomFunctions) {
      if (Entry.Pattern.match(*FuncDecl)) {
        StringRef Reason =
            Entry.Reason.empty() ? "is marked as unsafe" : Entry.Reason.c_str();

        if (Entry.Replacement.empty()) {
          diag(SourceExpr->getExprLoc(),
               "function %0 %1; it should not be used")
              << FuncDecl << Reason << Entry.Replacement
              << SourceExpr->getSourceRange();
        } else {
          diag(SourceExpr->getExprLoc(),
               "function %0 %1; '%2' should be used instead")
              << FuncDecl << Reason << Entry.Replacement
              << SourceExpr->getSourceRange();
        }

        return;
      }
    }

    llvm_unreachable("No custom function was matched.");
    return;
  }

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

  diag(SourceExpr->getExprLoc(), "function %0 %1; '%2' should be used instead")
      << FuncDecl << getRationaleFor(FunctionName)
      << ReplacementFunctionName.value() << SourceExpr->getSourceRange();
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
