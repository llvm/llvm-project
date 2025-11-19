//===--- UnsafeFormatStringCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeFormatStringCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/ConvertUTF.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

UnsafeFormatStringCheck::UnsafeFormatStringCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UnsafeFormatStringCheck::registerMatchers(MatchFinder *Finder) {
  // Matches sprintf and scanf family functions in std namespace in C++ and
  // globally in C.
  auto VulnerableFunctions =
      hasAnyName("sprintf", "vsprintf", "scanf", "fscanf", "sscanf", "vscanf",
                 "vfscanf", "vsscanf", "wscanf", "fwscanf", "swscanf",
                 "vwscanf", "vfwscanf", "vswscanf");
  Finder->addMatcher(
      callExpr(callee(functionDecl(VulnerableFunctions,
                                   anyOf(isInStdNamespace(),
                                         hasParent(translationUnitDecl())))),
               anyOf(hasArgument(0, stringLiteral().bind("format")),
                     hasArgument(1, stringLiteral().bind("format"))))
          .bind("call"),
      this);
}

void UnsafeFormatStringCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *Format = Result.Nodes.getNodeAs<StringLiteral>("format");

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

  const auto *Callee = cast<FunctionDecl>(Call->getCalleeDecl());
  StringRef FunctionName = Callee->getName();

  bool IsScanfFamily = FunctionName.contains("scanf");

  if (!hasUnboundedStringSpecifier(FormatString, IsScanfFamily))
    return;

  auto Diag =
      diag(
          Call->getBeginLoc(),
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
  size_t N = Fmt.size();
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
        if (!HasFieldWidth) {
          return true;
        }
      } else {
        // For sprintf family, only precision provides protection
        if (!HasPrecision) {
          return true;
        }
      }
    }

    Pos = SpecPos + 1;
  }

  return false;
}

} // namespace clang::tidy::bugprone
