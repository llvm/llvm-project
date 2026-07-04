//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InvalidRegexPatternCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void InvalidRegexPatternCheck::registerMatchers(MatchFinder *Finder) {
  // main matcher
  auto IsConstllvmStringRef = qualType(
      isConstQualified(), hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                              cxxRecordDecl(hasName("::llvm::StringRef"))))));
  auto IsConstStdString = qualType(
      isConstQualified(), hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                              cxxRecordDecl(hasName("::std::basic_string"))))));
  auto GetStringLit = ignoringImplicit(stringLiteral().bind("stringLiteral"));
  auto GetStringLiteralFromObject =
      ignoringImplicit(cxxConstructExpr(hasAnyArgument(GetStringLit)));
  auto IsConstCharPtr = pointerType(pointee(builtinType(), isConstQualified()));
  auto IsStdStringView = qualType(hasUnqualifiedDesugaredType(recordType(
      hasDeclaration(cxxRecordDecl(hasName("::std::basic_string_view"))))));
  auto AnyCastedToStringRef = ignoringImplicit(anyOf(
      stringLiteral().bind("stringLiteral"),
      declRefExpr(to(varDecl(hasType(IsConstStdString),
                             hasInitializer(GetStringLiteralFromObject)))),
      declRefExpr(to(varDecl(hasType(IsConstllvmStringRef),
                             hasInitializer(GetStringLiteralFromObject)))),
      declRefExpr(
          to(varDecl(hasType(IsConstCharPtr), hasInitializer(GetStringLit)))),
      declRefExpr(to(varDecl(hasType(IsStdStringView),
                             hasInitializer(GetStringLiteralFromObject)))),
      memberExpr(
          member(fieldDecl(hasType(IsConstStdString),
                           hasInClassInitializer(GetStringLiteralFromObject)))),
      memberExpr(member(fieldDecl(hasType(IsConstCharPtr),
                                  hasInClassInitializer(GetStringLit)))),
      memberExpr(
          member(fieldDecl(hasType(IsConstllvmStringRef),
                           hasInClassInitializer(GetStringLiteralFromObject)))),
      memberExpr(member(
          fieldDecl(hasType(IsStdStringView),
                    hasInClassInitializer(GetStringLiteralFromObject))))));

  auto IsRegexFlagsType = ignoringParenImpCasts(
      anyOf(integerLiteral().bind("regexFlagsInt"),
            declRefExpr(to(enumConstantDecl().bind("regexFlagEnum")))));
  Finder->addMatcher(
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(ofClass(hasName("llvm::Regex")))),
          hasArgument(0, ignoringImplicit(cxxConstructExpr(
                             hasDeclaration(cxxConstructorDecl(
                                 ofClass(hasName("::llvm::StringRef")))),
                             hasArgument(0, AnyCastedToStringRef)))),
          optionally(hasArgument(1, IsRegexFlagsType))),
      this);
}

void InvalidRegexPatternCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *DetectedPattern =
      Result.Nodes.getNodeAs<StringLiteral>("stringLiteral");
  if (DetectedPattern) {
    const auto *FlagInt =
        Result.Nodes.getNodeAs<IntegerLiteral>("regexFlagsInt");
    const auto *FlagEnum =
        Result.Nodes.getNodeAs<EnumConstantDecl>("regexFlagEnum");
    unsigned int Flag = llvm::Regex::RegexFlags::NoFlags;
    if (FlagInt)
      Flag = FlagInt->getValue().getZExtValue();
    if (FlagEnum)
      Flag = FlagEnum->getInitVal().getZExtValue();
    const llvm::Regex TestRegex(DetectedPattern->getString(), Flag);
    std::string RegexError;
    if (!TestRegex.isValid(RegexError))
      diag(DetectedPattern->getBeginLoc(), "invalid regex pattern: %0")
          << RegexError << DetectedPattern->getSourceRange();
  }
}

} // namespace clang::tidy::llvm_check
