//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypePromotionInMathFnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {
AST_MATCHER_P(Type, isBuiltinType, BuiltinType::Kind, Kind) {
  if (const auto *BT = dyn_cast<BuiltinType>(&Node)) {
    return BT->getKind() == Kind;
  }
  return false;
}
} // anonymous namespace

TypePromotionInMathFnCheck::TypePromotionInMathFnCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void TypePromotionInMathFnCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void TypePromotionInMathFnCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void TypePromotionInMathFnCheck::registerMatchers(MatchFinder *Finder) {
  constexpr BuiltinType::Kind IntTy = BuiltinType::Int;
  constexpr BuiltinType::Kind LongTy = BuiltinType::Long;
  constexpr BuiltinType::Kind FloatTy = BuiltinType::Float;
  constexpr BuiltinType::Kind DoubleTy = BuiltinType::Double;
  constexpr BuiltinType::Kind LongDoubleTy = BuiltinType::LongDouble;

  auto HasBuiltinTyParam = [](int Pos, BuiltinType::Kind Kind) {
    return hasParameter(Pos, hasType(isBuiltinType(Kind)));
  };
  auto HasBuiltinTyArg = [](int Pos, BuiltinType::Kind Kind) {
    return hasArgument(Pos, hasType(isBuiltinType(Kind)));
  };

  // Match calls to foo(double) with a float argument.
  auto OneDoubleArgFns = hasAnyName(
      "::acos", "::acosh", "::asin", "::asinh", "::atan", "::atanh", "::cbrt",
      "::ceil", "::cos", "::cosh", "::erf", "::erfc", "::exp", "::exp2",
      "::expm1", "::fabs", "::floor", "::ilogb", "::lgamma", "::llrint",
      "::log", "::log10", "::log1p", "::log2", "::logb", "::lrint", "::modf",
      "::nearbyint", "::rint", "::round", "::sin", "::sinh", "::sqrt", "::tan",
      "::tanh", "::tgamma", "::trunc", "::llround", "::lround");
  Finder->addMatcher(
      callExpr(callee(functionDecl(OneDoubleArgFns, parameterCountIs(1),
                                   HasBuiltinTyParam(0, DoubleTy))),
               HasBuiltinTyArg(0, FloatTy))
          .bind("call"),
      this);

  // Match calls to foo(double, double) where both args are floats.
  auto TwoDoubleArgFns = hasAnyName("::atan2", "::copysign", "::fdim", "::fmax",
                                    "::fmin", "::fmod", "::hypot", "::ldexp",
                                    "::nextafter", "::pow", "::remainder");
  Finder->addMatcher(
      callExpr(callee(functionDecl(TwoDoubleArgFns, parameterCountIs(2),
                                   HasBuiltinTyParam(0, DoubleTy),
                                   HasBuiltinTyParam(1, DoubleTy))),
               HasBuiltinTyArg(0, FloatTy), HasBuiltinTyArg(1, FloatTy))
          .bind("call"),
      this);

  // Match calls to fma(double, double, double) where all args are floats.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::fma"), parameterCountIs(3),
                                   HasBuiltinTyParam(0, DoubleTy),
                                   HasBuiltinTyParam(1, DoubleTy),
                                   HasBuiltinTyParam(2, DoubleTy))),
               HasBuiltinTyArg(0, FloatTy), HasBuiltinTyArg(1, FloatTy),
               HasBuiltinTyArg(2, FloatTy))
          .bind("call"),
      this);

  // Match calls to frexp(double, int*) where the first arg is a float.
  Finder->addMatcher(
      callExpr(callee(functionDecl(
                   hasName("::frexp"), parameterCountIs(2),
                   HasBuiltinTyParam(0, DoubleTy),
                   hasParameter(1, parmVarDecl(hasType(pointerType(
                                       pointee(isBuiltinType(IntTy)))))))),
               HasBuiltinTyArg(0, FloatTy))
          .bind("call"),
      this);

  // Match calls to nexttoward(double, long double) where the first arg is a
  // float.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::nexttoward"), parameterCountIs(2),
                                   HasBuiltinTyParam(0, DoubleTy),
                                   HasBuiltinTyParam(1, LongDoubleTy))),
               HasBuiltinTyArg(0, FloatTy))
          .bind("call"),
      this);

  // Match calls to remquo(double, double, int*) where the first two args are
  // floats.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(
              hasName("::remquo"), parameterCountIs(3),
              HasBuiltinTyParam(0, DoubleTy), HasBuiltinTyParam(1, DoubleTy),
              hasParameter(2, parmVarDecl(hasType(pointerType(
                                  pointee(isBuiltinType(IntTy)))))))),
          HasBuiltinTyArg(0, FloatTy), HasBuiltinTyArg(1, FloatTy))
          .bind("call"),
      this);

  // Match calls to scalbln(double, long) where the first arg is a float.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::scalbln"), parameterCountIs(2),
                                   HasBuiltinTyParam(0, DoubleTy),
                                   HasBuiltinTyParam(1, LongTy))),
               HasBuiltinTyArg(0, FloatTy))
          .bind("call"),
      this);

  // Match calls to scalbn(double, int) where the first arg is a float.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::scalbn"), parameterCountIs(2),
                                   HasBuiltinTyParam(0, DoubleTy),
                                   HasBuiltinTyParam(1, IntTy))),
               HasBuiltinTyArg(0, FloatTy))
          .bind("call"),
      this);

  // modf(double, double*) is omitted because the second parameter forces the
  // type -- there's no conversion from float* to double*.
}

void TypePromotionInMathFnCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  assert(Call != nullptr);

  StringRef OldFnName = Call->getDirectCallee()->getName();

  // In C++ mode, we prefer std::foo to ::foof.  But some of these suggestions
  // are only valid in C++11 and newer.
  static llvm::StringSet<> Cpp11OnlyFns = {
      "acosh",     "asinh",      "atanh",     "cbrt",   "copysign", "erf",
      "erfc",      "exp2",       "expm1",     "fdim",   "fma",      "fmax",
      "fmin",      "hypot",      "ilogb",     "lgamma", "llrint",   "llround",
      "log1p",     "log2",       "logb",      "lrint",  "lround",   "nearbyint",
      "nextafter", "nexttoward", "remainder", "remquo", "rint",     "round",
      "scalbln",   "scalbn",     "tgamma",    "trunc"};
  bool StdFnRequiresCpp11 = Cpp11OnlyFns.contains(OldFnName);

  std::string NewFnName;
  bool FnInCmath = false;
  if (getLangOpts().CPlusPlus &&
      (!StdFnRequiresCpp11 || getLangOpts().CPlusPlus11)) {
    NewFnName = ("std::" + OldFnName).str();
    FnInCmath = true;
  } else {
    NewFnName = (OldFnName + "f").str();
  }

  auto Diag = diag(Call->getExprLoc(), "call to '%0' promotes float to double")
              << OldFnName
              << FixItHint::CreateReplacement(
                     Call->getCallee()->getSourceRange(), NewFnName);

  // Suggest including <cmath> if the function we're suggesting is declared in
  // <cmath> and it's not already included.  We never have to suggest including
  // <math.h>, because the functions we're suggesting moving away from are all
  // declared in <math.h>.
  if (FnInCmath)
    Diag << IncludeInserter.createIncludeInsertion(
        Result.Context->getSourceManager().getFileID(Call->getBeginLoc()),
        "<cmath>");
}

} // namespace clang::tidy::performance
