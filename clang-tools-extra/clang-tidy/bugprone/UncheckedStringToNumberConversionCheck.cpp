//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedStringToNumberConversionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/FormatString.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringSwitch.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void UncheckedStringToNumberConversionCheck::registerMatchers(
    MatchFinder *Finder) {
  // Match any function call to the C standard library string conversion
  // functions that do no error checking.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(anyOf(
              functionDecl(hasAnyName("::atoi", "::atof", "::atol", "::atoll"))
                  .bind("converter"),
              functionDecl(hasAnyName("::scanf", "::sscanf", "::fscanf",
                                      "::vfscanf", "::vscanf", "::vsscanf"))
                  .bind("formatted")))))
          .bind("expr"),
      this);
}

namespace {
enum class ConversionKind {
  None,
  ToInt,
  ToUInt,
  ToLongInt,
  ToLongUInt,
  ToIntMax,
  ToUIntMax,
  ToFloat,
  ToDouble,
  ToLongDouble
};

} // namespace

static ConversionKind classifyConversionFunc(const FunctionDecl *FD) {
  return llvm::StringSwitch<ConversionKind>(FD->getName())
      .Cases("atoi", "atol", ConversionKind::ToInt)
      .Case("atoll", ConversionKind::ToLongInt)
      .Case("atof", ConversionKind::ToDouble)
      .Default(ConversionKind::None);
}

static ConversionKind classifyFormatString(StringRef Fmt, const LangOptions &LO,
                                           const TargetInfo &TI) {
  // Scan the format string for the first problematic format specifier, then
  // report that as the conversion type. This will miss additional conversion
  // specifiers, but that is acceptable behavior.

  class Handler : public analyze_format_string::FormatStringHandler {
    ConversionKind CK = ConversionKind::None;

    bool HandleScanfSpecifier(const analyze_scanf::ScanfSpecifier &FS,
                              const char *StartSpecifier,
                              unsigned SpecifierLen) override {
      // If we just consume the argument without assignment, we don't care
      // about it having conversion errors.
      if (!FS.consumesDataArgument())
        return true;

      // Get the conversion specifier and use it to determine the conversion
      // kind.
      analyze_scanf::ScanfConversionSpecifier SCS = FS.getConversionSpecifier();
      if (SCS.isIntArg()) {
        switch (FS.getLengthModifier().getKind()) {
        case analyze_scanf::LengthModifier::AsLongLong:
          CK = ConversionKind::ToLongInt;
          break;
        case analyze_scanf::LengthModifier::AsIntMax:
          CK = ConversionKind::ToIntMax;
          break;
        default:
          CK = ConversionKind::ToInt;
          break;
        }
      } else if (SCS.isUIntArg()) {
        switch (FS.getLengthModifier().getKind()) {
        case analyze_scanf::LengthModifier::AsLongLong:
          CK = ConversionKind::ToLongUInt;
          break;
        case analyze_scanf::LengthModifier::AsIntMax:
          CK = ConversionKind::ToUIntMax;
          break;
        default:
          CK = ConversionKind::ToUInt;
          break;
        }
      } else if (SCS.isDoubleArg()) {
        switch (FS.getLengthModifier().getKind()) {
        case analyze_scanf::LengthModifier::AsLongDouble:
          CK = ConversionKind::ToLongDouble;
          break;
        case analyze_scanf::LengthModifier::AsLong:
          CK = ConversionKind::ToDouble;
          break;
        default:
          CK = ConversionKind::ToFloat;
          break;
        }
      }

      // Continue if we have yet to find a conversion kind that we care about.
      return CK == ConversionKind::None;
    }

  public:
    Handler() = default;

    ConversionKind get() const { return CK; }
  };

  Handler H;
  analyze_format_string::ParseScanfString(H, Fmt.begin(), Fmt.end(), LO, TI);

  return H.get();
}

static StringRef classifyConversionType(ConversionKind K) {
  switch (K) {
  case ConversionKind::None:
    llvm_unreachable("Unexpected conversion kind");
  case ConversionKind::ToInt:
  case ConversionKind::ToLongInt:
  case ConversionKind::ToIntMax:
    return "an integer value";
  case ConversionKind::ToUInt:
  case ConversionKind::ToLongUInt:
  case ConversionKind::ToUIntMax:
    return "an unsigned integer value";
  case ConversionKind::ToFloat:
  case ConversionKind::ToDouble:
  case ConversionKind::ToLongDouble:
    return "a floating-point value";
  }
  llvm_unreachable("Unknown conversion kind");
}

static StringRef classifyReplacement(ConversionKind K) {
  switch (K) {
  case ConversionKind::None:
    llvm_unreachable("Unexpected conversion kind");
  case ConversionKind::ToInt:
    return "strtol";
  case ConversionKind::ToUInt:
    return "strtoul";
  case ConversionKind::ToIntMax:
    return "strtoimax";
  case ConversionKind::ToLongInt:
    return "strtoll";
  case ConversionKind::ToLongUInt:
    return "strtoull";
  case ConversionKind::ToUIntMax:
    return "strtoumax";
  case ConversionKind::ToFloat:
    return "strtof";
  case ConversionKind::ToDouble:
    return "strtod";
  case ConversionKind::ToLongDouble:
    return "strtold";
  }
  llvm_unreachable("Unknown conversion kind");
}

void UncheckedStringToNumberConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("expr");
  const FunctionDecl *FuncDecl = nullptr;
  ConversionKind Conversion = ConversionKind::None;

  if (const auto *ConverterFunc =
          Result.Nodes.getNodeAs<FunctionDecl>("converter")) {
    // Converter functions are always incorrect to use.
    FuncDecl = ConverterFunc;
    Conversion = classifyConversionFunc(ConverterFunc);
  } else if (const auto *FFD =
                 Result.Nodes.getNodeAs<FunctionDecl>("formatted")) {
    StringRef FmtStr;
    // The format string comes from the call expression and depends on which
    // flavor of scanf is called.
    // Index 0: scanf, vscanf, Index 1: fscanf, sscanf, vfscanf, vsscanf.
    unsigned Idx =
        (FFD->getName() == "scanf" || FFD->getName() == "vscanf") ? 0 : 1;

    // Given the index, see if the call expression argument at that index is
    // a string literal.
    if (Call->getNumArgs() < Idx)
      return;

    if (const Expr *Arg = Call->getArg(Idx)->IgnoreParenImpCasts()) {
      if (const auto *SL = dyn_cast<StringLiteral>(Arg)) {
        FmtStr = SL->getString();
      }
    }

    // If we could not get the format string, bail out.
    if (FmtStr.empty())
      return;

    // Formatted input functions need further checking of the format string to
    // determine whether a problematic conversion may be happening.
    Conversion = classifyFormatString(FmtStr, getLangOpts(),
                                      Result.Context->getTargetInfo());
    if (Conversion != ConversionKind::None)
      FuncDecl = FFD;
  }

  if (!FuncDecl)
    return;

  diag(Call->getExprLoc(),
       "%0 used to convert a string to %1, but function will not report "
       "conversion errors; consider using '%2' instead")
      << FuncDecl << classifyConversionType(Conversion)
      << classifyReplacement(Conversion);
}

} // namespace clang::tidy::bugprone
