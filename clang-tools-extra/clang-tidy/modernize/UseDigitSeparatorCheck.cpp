//===--- UseDigitSeparatorCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <numeric>
#include <regex>

#include "UseDigitSeparatorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang::ast_matchers;

namespace {
std::string
getFormatedScientificFloatString(const llvm::StringRef OriginalLiteralString);

std::vector<std::string> splitStringByGroupSize(const std::string &String,
                                                size_t GroupSize) {
  std::vector<std::string> Result;
  std::string ReversedString(String.rbegin(), String.rend());

  for (size_t I = 0; I < ReversedString.size(); I += GroupSize) {
    Result.push_back(ReversedString.substr(I, GroupSize));
  }

  std::reverse(Result.begin(), Result.end());
  std::for_each(Result.begin(), Result.end(), [](std::string &Str) {
    return std::reverse(Str.begin(), Str.end());
  });

  return Result;
}

std::string
getFormatedIntegerString(const llvm::StringRef OriginalLiteralString,
                         const llvm::APInt IntegerValue) {
  // Configure formatting
  unsigned int Radix;
  size_t GroupSize;
  std::string Prefix;
  std::string Postfix;
  if (OriginalLiteralString.starts_with("0b")) {
    Radix = 2;
    GroupSize = 4;
    Prefix = "0b";
  } else if (OriginalLiteralString.starts_with("0x")) {
    Radix = 16;
    GroupSize = 4;
    Prefix = "0x";
  } else if (OriginalLiteralString.starts_with("0") &&
             OriginalLiteralString != "0") {
    Radix = 8;
    GroupSize = 3;
    Prefix = "0";
  } else {
    Radix = 10;
    GroupSize = 3;
  }

  for (const char &Character : OriginalLiteralString.substr(
           Prefix.size(), OriginalLiteralString.size())) {
    if (((Prefix == "0x" && !std::isxdigit(Character)) ||
         (Prefix != "0x" && !std::isdigit(Character))) &&
        Character != '\'') {
      Postfix += Character;
    }
  }

  // Get formatting literal text
  const std::vector<std::string> SplittedIntegerLiteral =
      splitStringByGroupSize(toString(IntegerValue, Radix, true), GroupSize);
  const std::string FormatedLiteralString =
      Prefix +
      std::accumulate(
          SplittedIntegerLiteral.begin(), SplittedIntegerLiteral.end(),
          std::string(""),
          [](std::string S1, std::string S2) { return S1 + "\'" + S2; })
          .erase(0, 1) +
      Postfix;

  return FormatedLiteralString;
}

std::string getFormatedFloatString(const llvm::StringRef OriginalLiteralString,
                                   const llvm::APFloat FloatValue) {
  if (OriginalLiteralString.contains('E') ||
      OriginalLiteralString.contains('e')) {
    return getFormatedScientificFloatString(OriginalLiteralString);
  }

  // Configure formatting and get precision
  std::string Postfix;
  int Precision = 0;
  for (const char &Character : OriginalLiteralString) {
    if (!std::isdigit(Character) && Character != '.' && Character != '\'') {
      Postfix += Character;
    } else if (std::isdigit(Character)) {
      Precision++;
    }
  }

  // Get formatting literal text

  // Get string representation of float value
  llvm::SmallString<128> FloatString;
  FloatValue.toString(FloatString, Precision);

  // Get integer and fractional parts of float number
  const std::string::size_type DotPosition = FloatString.find('.');
  const llvm::SmallString<128> IntegerSubString =
      FloatString.substr(0, DotPosition);
  llvm::SmallString<128> FractionalSubString =
      FloatString.substr(DotPosition + 1, FloatString.size());
  std::reverse(FractionalSubString.begin(), FractionalSubString.end());

  // Get formatting literal text
  const std::string FormatedIntegerSubString = getFormatedIntegerString(
      IntegerSubString,
      llvm::APInt(128, std::stoll(IntegerSubString.str().str())));
  std::string FormatedFractionalSubString = getFormatedIntegerString(
      FractionalSubString,
      llvm::APInt(128, std::stoll(FractionalSubString.str().str())));
  std::reverse(FormatedFractionalSubString.begin(),
               FormatedFractionalSubString.end());

  const std::string FormatedLiteralString =
      FormatedIntegerSubString + '.' + FormatedFractionalSubString + Postfix;

  return FormatedLiteralString;
}

std::string
getFormatedScientificFloatString(const llvm::StringRef OriginalLiteralString) {
  // Split string to mantissa and exponent
  const char EChar =
      OriginalLiteralString.str().find('E') != std::string::npos ? 'E' : 'e';
  const std::string::size_type EPosition =
      OriginalLiteralString.str().find(EChar);
  const std::string SignSymbol =
      OriginalLiteralString[EPosition + 1] == '-'
          ? "-"
          : (OriginalLiteralString[EPosition + 1] == '+' ? "+" : "");
  const llvm::StringRef MantissaSubString =
      OriginalLiteralString.substr(0, EPosition);
  const llvm::StringRef ExponentSubString = OriginalLiteralString.substr(
      EPosition + SignSymbol.size() + 1, OriginalLiteralString.size());
  std::string MantissaSubStringWithoutBraces = MantissaSubString.str();
  MantissaSubStringWithoutBraces.erase(
      std::remove_if(MantissaSubStringWithoutBraces.begin(),
                     MantissaSubStringWithoutBraces.end(),
                     [](unsigned char Char) { return Char == '\''; }),
      MantissaSubStringWithoutBraces.end());
  std::string ExponentSubStringWithoutBraces = ExponentSubString.str();
  ExponentSubStringWithoutBraces.erase(
      std::remove_if(ExponentSubStringWithoutBraces.begin(),
                     ExponentSubStringWithoutBraces.end(),
                     [](unsigned char Char) { return Char == '\''; }),
      ExponentSubStringWithoutBraces.end());

  // Get formatting literal text
  const std::string FormatedMantissaString = getFormatedFloatString(
      MantissaSubString,
      llvm::APFloat(std::stod(MantissaSubStringWithoutBraces)));
  const std::string FormatedExponentString = getFormatedIntegerString(
      ExponentSubString,
      llvm::APInt(128, std::stoll(ExponentSubStringWithoutBraces)));
  return FormatedMantissaString + EChar + SignSymbol + FormatedExponentString;
}
} // namespace

namespace clang::tidy::modernize {

void UseDigitSeparatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(integerLiteral().bind("integerLiteral"), this);
  Finder->addMatcher(floatLiteral().bind("floatLiteral"), this);
}

void UseDigitSeparatorCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();
  const IntegerLiteral *MatchedInteger =
      Result.Nodes.getNodeAs<IntegerLiteral>("integerLiteral");
  const FloatingLiteral *MatchedFloat =
      Result.Nodes.getNodeAs<FloatingLiteral>("floatLiteral");

  if (MatchedInteger != nullptr) {
    // Get original literal source text
    const llvm::StringRef OriginalLiteralString = Lexer::getSourceText(
        CharSourceRange::getTokenRange(MatchedInteger->getSourceRange()),
        Source, Context.getLangOpts());

    // Get formatting literal text
    const llvm::APInt IntegerValue = MatchedInteger->getValue();
    const std::string FormatedLiteralString =
        getFormatedIntegerString(OriginalLiteralString, IntegerValue);

    // Compare the original and formatted representation of a literal
    if (OriginalLiteralString != FormatedLiteralString) {
      diag(MatchedInteger->getLocation(),
           "unformatted representation of integer literal '%0'")
          << OriginalLiteralString
          << FixItHint::CreateReplacement(MatchedInteger->getLocation(),
                                          FormatedLiteralString);
    }

    return;
  }

  if (MatchedFloat != nullptr) {
    // Get original literal source text
    const llvm::StringRef OriginalLiteralString = Lexer::getSourceText(
        CharSourceRange::getTokenRange(MatchedFloat->getSourceRange()), Source,
        Context.getLangOpts());

    // Get formatting literal text
    const llvm::APFloat FloatValue = MatchedFloat->getValue();
    const std::string FormatedLiteralString =
        getFormatedFloatString(OriginalLiteralString, FloatValue);

    // Compare the original and formatted representation of a literal
    if (OriginalLiteralString != FormatedLiteralString) {
      diag(MatchedFloat->getLocation(),
           "unformatted representation of floating literal '%0'")
          << OriginalLiteralString
          << FixItHint::CreateReplacement(MatchedFloat->getLocation(),
                                          FormatedLiteralString);
    }

    return;
  }

  llvm_unreachable("Unreachable code in UseDigitSeparatorCheck");
}

class UseDigitSeparatorCallbacks : public PPCallbacks {
public:
  UseDigitSeparatorCallbacks(UseDigitSeparatorCheck *Check) : Check(Check) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;

private:
  UseDigitSeparatorCheck *Check;
};
void UseDigitSeparatorCallbacks::MacroDefined(const Token &MacroNameTok,
                                              const MacroDirective *MD) {
  const MacroInfo *Info = MD->getMacroInfo();
  ArrayRef<Token> MacroTokens = Info->tokens();
  if (Info->isBuiltinMacro() || MacroTokens.empty()) {
    return;
  }
  for (const Token &T : MacroTokens) {
    if (!T.isLiteral()) {
      continue;
    }

    // Get original literal source text
    std::string OriginalLiteralString(T.getLiteralData(), T.getLength());

    if (!isDigit(OriginalLiteralString[0])) {
      return;
    }

    // Get formatting literal text
    llvm::APInt IntegerValue;
    if (llvm::StringRef(OriginalLiteralString).starts_with("0b")) {
      std::string BinaryNumber = OriginalLiteralString;
      BinaryNumber.erase(0, 2);
      IntegerValue = llvm::APInt(128, std::stoul(BinaryNumber, nullptr, 2));
    } else if (llvm::StringRef(OriginalLiteralString).starts_with("0x")) {
      IntegerValue =
          llvm::APInt(128, std::stoul(OriginalLiteralString, nullptr, 16));
    } else if (llvm::StringRef(OriginalLiteralString).starts_with("0")) {
      IntegerValue =
          llvm::APInt(128, std::stoul(OriginalLiteralString, nullptr, 8));
    } else {
      IntegerValue = llvm::APInt(128, std::stoul(OriginalLiteralString));
    }
    const std::string FormatedLiteralString =
        getFormatedIntegerString(OriginalLiteralString, IntegerValue);

    if (OriginalLiteralString != FormatedLiteralString) {
      Check->diag(T.getLocation(),
                  "unformatted representation of integer literal '%0'")
          << OriginalLiteralString
          << FixItHint::CreateReplacement(T.getLocation(),
                                          FormatedLiteralString);
    }
  }
}

void UseDigitSeparatorCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  auto Callback = std::make_unique<UseDigitSeparatorCallbacks>(this);
  PP->addPPCallbacks(std::move(Callback));
}

} // namespace clang::tidy::modernize
