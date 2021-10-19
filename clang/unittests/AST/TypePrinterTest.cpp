//===- unittests/AST/TypePrinterTest.cpp --- Type printer tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for QualType::print() and related methods.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

static void PrintType(raw_ostream &Out, const ASTContext *Context,
                      const QualType *T,
                      PrintingPolicyAdjuster PolicyAdjuster) {
  assert(T && !T->isNull() && "Expected non-null Type");
  PrintingPolicy Policy = Context->getPrintingPolicy();
  if (PolicyAdjuster)
    PolicyAdjuster(Policy);
  T->print(Out, Policy);
}

::testing::AssertionResult
PrintedTypeMatches(StringRef Code, const std::vector<std::string> &Args,
                   const DeclarationMatcher &NodeMatch,
                   StringRef ExpectedPrinted,
                   PrintingPolicyAdjuster PolicyAdjuster) {
  return PrintedNodeMatches<QualType>(Code, Args, NodeMatch, ExpectedPrinted,
                                      "", PrintType, PolicyAdjuster);
}

} // unnamed namespace

TEST(TypePrinter, TemplateId) {
  std::string Code = R"cpp(
    namespace N {
      template <typename> struct Type {};
      
      template <typename T>
      void Foo(const Type<T> &Param);
    }
  )cpp";
  auto Matcher = parmVarDecl(hasType(qualType().bind("id")));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "const Type<T> &",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = false; }));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "const N::Type<T> &",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = true; }));
}