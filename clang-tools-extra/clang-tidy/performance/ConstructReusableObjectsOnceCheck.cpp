//===--- ConstructReusableObjectsOnceCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstructReusableObjectsOnceCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

const llvm::StringRef DefaultCheckedClasses =
    "::std::basic_regex;::boost::basic_regex";
const llvm::StringRef DefaultIgnoredFunctions = "::main";

} // namespace

ConstructReusableObjectsOnceCheck::ConstructReusableObjectsOnceCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckedClasses(utils::options::parseStringList(
          Options.get("CheckedClasses", DefaultCheckedClasses))),
      IgnoredFunctions(utils::options::parseStringList(
          Options.get("IgnoredFunctions", DefaultIgnoredFunctions))) {}

void ConstructReusableObjectsOnceCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckedClasses", DefaultCheckedClasses);
  Options.store(Opts, "IgnoredFunctions", DefaultIgnoredFunctions);
}

void ConstructReusableObjectsOnceCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConstStrLiteralDecl =
      varDecl(unless(parmVarDecl()), hasType(constantArrayType()),
              hasType(isConstQualified()),
              hasInitializer(ignoringParenImpCasts(stringLiteral())));
  const auto ConstPtrStrLiteralDecl = varDecl(
      unless(parmVarDecl()),
      hasType(pointerType(pointee(isAnyCharacter(), isConstQualified()))),
      hasInitializer(ignoringParenImpCasts(stringLiteral())));

  const auto ConstNumberLiteralDecl =
      varDecl(hasType(qualType(anyOf(isInteger(), realFloatingPointType()))),
              hasType(isConstQualified()),
              hasInitializer(ignoringParenImpCasts(
                  anyOf(integerLiteral(), floatLiteral()))),
              unless(parmVarDecl()));

  const auto ConstEnumLiteralDecl = varDecl(
      unless(parmVarDecl()), hasType(hasUnqualifiedDesugaredType(enumType())),
      hasType(isConstQualified()),
      hasInitializer(declRefExpr(to(enumConstantDecl()))));

  const auto ConstLiteralArg = expr(ignoringParenImpCasts(
      anyOf(stringLiteral(), integerLiteral(), floatLiteral(),
            declRefExpr(to(enumConstantDecl())),
            declRefExpr(hasDeclaration(
                anyOf(ConstNumberLiteralDecl, ConstPtrStrLiteralDecl,
                      ConstStrLiteralDecl, ConstEnumLiteralDecl))))));

  const auto ConstructorCall = cxxConstructExpr(
      hasDeclaration(cxxConstructorDecl(
          ofClass(cxxRecordDecl(hasAnyName(CheckedClasses)).bind("class")))),
      unless(hasAnyArgument(expr(unless(ConstLiteralArg)))));

  Finder->addMatcher(
      varDecl(unless(hasGlobalStorage()), hasInitializer(ConstructorCall),
              hasAncestor(functionDecl(unless(hasAnyName(IgnoredFunctions)))
                              .bind("function")))
          .bind("var"),
      this);
}

void ConstructReusableObjectsOnceCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    const auto *Class = Result.Nodes.getNodeAs<CXXRecordDecl>("class");
    assert(Class);

    const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
    assert(Function);

    diag(Var->getLocation(),
         "variable '%0' of type '%1' is constructed with only constant "
         "literals on each invocation of '%2'; make this variable 'static', "
         "declare as a global variable or move to a class member to avoid "
         "repeated constructions")
        << Var->getName() << Class->getQualifiedNameAsString()
        << Function->getQualifiedNameAsString();
  }
}

} // namespace clang::tidy::performance
