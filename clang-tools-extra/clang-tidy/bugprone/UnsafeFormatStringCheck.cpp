//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeFormatStringCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static constexpr llvm::StringRef OptionNameCustomPrintfFunctions =
    "CustomPrintfFunctions";
static constexpr llvm::StringRef OptionNameCustomScanfFunctions =
    "CustomScanfFunctions";

static constexpr llvm::StringRef BuiltInFormatBind = "format";
static constexpr llvm::StringRef BuiltInCallBind = "call";
static constexpr llvm::StringRef PrintfCallBind = "printfcall";
static constexpr llvm::StringRef ScanfCallBind = "scanfcall";

static std::vector<UnsafeFormatStringCheck::CheckedFunction>
parseCheckedFunctions(StringRef Option, ClangTidyContext *Context) {
  const std::vector<StringRef> Functions =
      utils::options::parseStringList(Option);
  std::vector<UnsafeFormatStringCheck::CheckedFunction> Result;
  Result.reserve(Functions.size());

  for (const StringRef Function : Functions) {
    if (Function.empty())
      continue;
    const auto [Name, ParamCount] = Function.split(',');
    unsigned long Count = 0;
    if (Name.trim().empty() || ParamCount.trim().empty() ||
        ParamCount.trim().getAsInteger(10, Count)) {
      Context->configurationDiag(
          "invalid configuration value for option '%0'; "
          "expected <functionname>, <paramcount>; pairs.")
          << OptionNameCustomPrintfFunctions;
      continue;
    }
    Result.push_back(
        {Name.trim().str(),
         matchers::MatchesAnyListedRegexNameMatcher::NameMatcher(Name.trim()),
         Count});
  }

  return Result;
}

UnsafeFormatStringCheck::UnsafeFormatStringCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CustomPrintfFunctions(parseCheckedFunctions(
          Options.get(OptionNameCustomPrintfFunctions, ""), Context)),
      CustomScanfFunctions(parseCheckedFunctions(
          Options.get(OptionNameCustomScanfFunctions, ""), Context)) {}

void UnsafeFormatStringCheck::registerMatchers(MatchFinder *Finder) {
  // Matches sprintf and scanf family functions in std namespace in C++ and
  // globally in C.
  const auto VulnerableFunctionsArg0 =
      hasAnyName("scanf", "vscanf", "wscanf", "vwscanf");
  const auto VulnerableFunctionsArg1 =
      hasAnyName("sprintf", "vsprintf", "fscanf", "sscanf", "vscanf", "vfscanf",
                 "vsscanf", "fwscanf", "swscanf", "vfwscanf", "vswscanf");
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(VulnerableFunctionsArg0,
                              anyOf(isInStdNamespace(),
                                    hasDeclContext(translationUnitDecl())))),
          hasArgument(0, stringLiteral().bind(BuiltInFormatBind)))
          .bind(BuiltInCallBind),
      this);

  Finder->addMatcher(
      callExpr(
          callee(functionDecl(VulnerableFunctionsArg1,
                              anyOf(isInStdNamespace(),
                                    hasDeclContext(translationUnitDecl())))),
          hasArgument(1, stringLiteral().bind(BuiltInFormatBind)))
          .bind(BuiltInCallBind),
      this);

  if (!CustomPrintfFunctions.empty()) {
    std::vector<llvm::StringRef> FunctionNames;
    FunctionNames.reserve(CustomPrintfFunctions.size());

    for (const CheckedFunction &Entry : CustomPrintfFunctions)
      FunctionNames.emplace_back(Entry.Name);

    const auto CustomFunctionsMatcher =
        matchers::matchesAnyListedRegexName(FunctionNames);

    Finder->addMatcher(callExpr(callee(functionDecl(CustomFunctionsMatcher)))
                           .bind(PrintfCallBind),
                       this);
  }

  if (!CustomScanfFunctions.empty()) {
    std::vector<llvm::StringRef> FunctionNames;
    FunctionNames.reserve(CustomScanfFunctions.size());

    for (const CheckedFunction &Entry : CustomScanfFunctions)
      FunctionNames.emplace_back(Entry.Name);

    const auto CustomFunctionsMatcher =
        matchers::matchesAnyListedRegexName(FunctionNames);

    Finder->addMatcher(callExpr(callee(functionDecl(CustomFunctionsMatcher)))
                           .bind(ScanfCallBind),
                       this);
  }
}

void UnsafeFormatStringCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, OptionNameCustomPrintfFunctions, "");
  Options.store(Opts, OptionNameCustomScanfFunctions, "");
}

const StringLiteral *UnsafeFormatStringCheck::getFormatLiteral(
    const CallExpr *Call, const std::vector<CheckedFunction> &CustomFunctions) {
  const auto *FD = Call->getDirectCallee();
  if (!FD)
    return nullptr;
  for (const CheckedFunction &Entry : CustomFunctions) {
    if (Entry.Pattern.match(*FD)) {
      if (Entry.FormatStringLocation >= Call->getNumArgs())
        return nullptr;
      const Expr *Arg =
          Call->getArg(Entry.FormatStringLocation)->IgnoreImpCasts();
      return dyn_cast<StringLiteral>(Arg);
    }
  }
  return nullptr;
}

void UnsafeFormatStringCheck::check(const MatchFinder::MatchResult &Result) {
  const CallExpr *Call = nullptr;
  const StringLiteral *Format = nullptr;
  bool IsScanfFamily = false;
  if ((Call = Result.Nodes.getNodeAs<CallExpr>(BuiltInCallBind))) {
    Format = Result.Nodes.getNodeAs<StringLiteral>(BuiltInFormatBind);
    const auto *Callee = cast<FunctionDecl>(Call->getCalleeDecl());
    const StringRef FunctionName = Callee->getName();
    IsScanfFamily = FunctionName.contains("scanf");
  } else if ((Call = Result.Nodes.getNodeAs<CallExpr>(PrintfCallBind))) {
    Format =
        UnsafeFormatStringCheck::getFormatLiteral(Call, CustomPrintfFunctions);
    IsScanfFamily = false;
  } else if ((Call = Result.Nodes.getNodeAs<CallExpr>(ScanfCallBind))) {
    Format =
        UnsafeFormatStringCheck::getFormatLiteral(Call, CustomScanfFunctions);
    IsScanfFamily = true;
  } else {
    Call = nullptr;
    Format = nullptr;
    llvm_unreachable("No valid matched node in check()");
  }

  if (!Call || !Format)
    return;

  std::string FormatString;
  if (Format->getCharByteWidth() == 1) {
    FormatString = Format->getString().str();
  } else if (Format->getCharByteWidth() == 2) {
    // Handle wide strings by converting to narrow string for analysis
    convertUTF16ToUTF8String(Format->getBytes(), FormatString);
  } else if (Format->getCharByteWidth() == 4) {
    // Handle wide strings by converting to narrow string for analysis
    convertUTF32ToUTF8String(Format->getBytes(), FormatString);
  }

  if (!hasUnboundedStringSpecifier(FormatString, IsScanfFamily))
    return;

  diag(Call->getBeginLoc(),
       IsScanfFamily
           ? "format specifier '%%s' without field width may cause buffer "
             "overflow; consider using '%%Ns' where N limits input length"
           : "format specifier '%%s' without precision may cause buffer "
             "overflow; consider using '%%.Ns' where N limits output length")
      << Call->getSourceRange();
}

bool UnsafeFormatStringCheck::hasUnboundedStringSpecifier(StringRef Fmt,
                                                          bool IsScanfFamily) {
  size_t Pos = 0;
  const size_t N = Fmt.size();
  while ((Pos = Fmt.find('%', Pos)) != StringRef::npos) {
    if (Pos + 1 >= N)
      break;

    // Skip %%
    if (Fmt[Pos + 1] == '%') {
      Pos += 2;
      continue;
    }

    size_t SpecPos = Pos + 1;

    // Skip flags
    while (SpecPos < N &&
           (Fmt[SpecPos] == '-' || Fmt[SpecPos] == '+' || Fmt[SpecPos] == ' ' ||
            Fmt[SpecPos] == '#' || Fmt[SpecPos] == '0')) {
      SpecPos++;
    }

    // Check for field width
    bool HasFieldWidth = false;
    if (SpecPos < N && Fmt[SpecPos] == '*') {
      HasFieldWidth = true;
      SpecPos++;
    } else {
      while (SpecPos < N && isdigit(Fmt[SpecPos])) {
        HasFieldWidth = true;
        SpecPos++;
      }
    }

    // Check for precision
    bool HasPrecision = false;
    if (SpecPos < N && Fmt[SpecPos] == '.') {
      SpecPos++;
      if (SpecPos < N && Fmt[SpecPos] == '*') {
        HasPrecision = true;
        SpecPos++;
      } else {
        while (SpecPos < N && isdigit(Fmt[SpecPos])) {
          HasPrecision = true;
          SpecPos++;
        }
      }
    }

    // Skip length modifiers
    while (SpecPos < N && (Fmt[SpecPos] == 'h' || Fmt[SpecPos] == 'l' ||
                           Fmt[SpecPos] == 'L' || Fmt[SpecPos] == 'z' ||
                           Fmt[SpecPos] == 'j' || Fmt[SpecPos] == 't')) {
      SpecPos++;
    }

    // Check for 's' specifier
    if (SpecPos < N && Fmt[SpecPos] == 's') {
      if (IsScanfFamily) {
        // For scanf family, field width provides protection
        if (!HasFieldWidth)
          return true;
      } else {
        // For sprintf family, only precision provides protection
        if (!HasPrecision)
          return true;
      }
    }

    Pos = SpecPos + 1;
  }

  return false;
}

} // namespace clang::tidy::bugprone
