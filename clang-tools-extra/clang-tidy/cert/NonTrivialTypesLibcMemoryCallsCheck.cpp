//===--- NonTrivialTypesLibcMemoryCallsCheck.cpp - clang-tidy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NonTrivialTypesLibcMemoryCallsCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

namespace {
AST_MATCHER(CXXRecordDecl, isTriviallyDefaultConstructible) {
  return Node.hasTrivialDefaultConstructor();
}
AST_MATCHER(CXXRecordDecl, isTriviallyCopyable) {
  return Node.hasTrivialCopyAssignment() && Node.hasTrivialCopyConstructor();
}
} // namespace

static const char BuiltinMemSet[] = "::std::memset;"
                                    "::memset;";
static const char BuiltinMemCpy[] = "::std::memcpy;"
                                    "::memcpy;"
                                    "::std::memmove;"
                                    "::memmove;"
                                    "::std::strcpy;"
                                    "::strcpy;"
                                    "::memccpy;"
                                    "::stpncpy;"
                                    "::strncpy;";
static const char BuiltinMemCmp[] = "::std::memcmp;"
                                    "::memcmp;"
                                    "::std::strcmp;"
                                    "::strcmp;"
                                    "::strncmp;";
static constexpr llvm::StringRef ComparisonOperators[] = {
    "operator==", "operator!=", "operator<",
    "operator>",  "operator<=", "operator>="};

NonTrivialTypesLibcMemoryCallsCheck::NonTrivialTypesLibcMemoryCallsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MemSetNames(Options.get("MemSetNames", "")),
      MemCpyNames(Options.get("MemCpyNames", "")),
      MemCmpNames(Options.get("MemCmpNames", "")) {}

void NonTrivialTypesLibcMemoryCallsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MemSetNames", MemSetNames);
  Options.store(Opts, "MemCpyNames", MemCpyNames);
  Options.store(Opts, "MemCmpNames", MemCmpNames);
}

void NonTrivialTypesLibcMemoryCallsCheck::registerMatchers(
    MatchFinder *Finder) {
  using namespace ast_matchers::internal;
  auto IsStructPointer = [](Matcher<CXXRecordDecl> Constraint = anything(),
                            bool Bind = false) {
    return expr(unaryOperator(
        hasOperatorName("&"),
        hasUnaryOperand(declRefExpr(
            hasType(cxxRecordDecl(Constraint)),
            hasType(Bind ? qualType().bind("Record") : qualType())))));
  };
  auto IsRecordSizeOf =
      expr(sizeOfExpr(hasArgumentOfType(equalsBoundNode("Record"))));
  auto ArgChecker = [&](Matcher<CXXRecordDecl> RecordConstraint,
                        BindableMatcher<Stmt> SecondArg = expr()) {
    return allOf(argumentCountIs(3),
                 hasArgument(0, IsStructPointer(RecordConstraint, true)),
                 hasArgument(1, SecondArg), hasArgument(2, IsRecordSizeOf));
  };

  Finder->addMatcher(
      callExpr(callee(namedDecl(hasAnyName(
                   utils::options::parseListPair(BuiltinMemSet, MemSetNames)))),
               ArgChecker(unless(isTriviallyDefaultConstructible())))
          .bind("lazyConstruct"),
      this);
  Finder->addMatcher(
      callExpr(callee(namedDecl(hasAnyName(
                   utils::options::parseListPair(BuiltinMemCpy, MemCpyNames)))),
               ArgChecker(unless(isTriviallyCopyable()), IsStructPointer()))
          .bind("lazyCopy"),
      this);
  Finder->addMatcher(
      callExpr(callee(namedDecl(hasAnyName(
                   utils::options::parseListPair(BuiltinMemCmp, MemCmpNames)))),
               ArgChecker(hasMethod(hasAnyName(ComparisonOperators)),
                          IsStructPointer()))
          .bind("lazyCompare"),
      this);
}

void NonTrivialTypesLibcMemoryCallsCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Caller = Result.Nodes.getNodeAs<CallExpr>("lazyConstruct")) {
    diag(Caller->getBeginLoc(), "calling %0 on a non-trivially default "
                                "constructible class is undefined")
        << cast<NamedDecl>(Caller->getCalleeDecl());
  }
  if (const auto *Caller = Result.Nodes.getNodeAs<CallExpr>("lazyCopy")) {
    diag(Caller->getBeginLoc(),
         "calling %0 on a non-trivially copyable class is undefined")
        << cast<NamedDecl>(Caller->getCalleeDecl());
  }
  if (const auto *Caller = Result.Nodes.getNodeAs<CallExpr>("lazyCompare")) {
    diag(Caller->getBeginLoc(),
         "consider using comparison operators instead of calling %0")
        << cast<NamedDecl>(Caller->getCalleeDecl());
  }
}

} // namespace clang::tidy::cert
