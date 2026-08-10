//===- unittests/AST/ConceptPrinterTest.cpp --- Concept printer tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

static void PrintConceptReference(raw_ostream &Out, const ASTContext *Context,
                                  const ConceptSpecializationExpr *T,
                                  PrintingPolicyAdjuster PolicyAdjuster) {
  assert(T && T->getConceptReference() &&
         "Expected non-null concept reference");

  PrintingPolicy Policy = Context->getPrintingPolicy();
  if (PolicyAdjuster)
    PolicyAdjuster(Policy);
  T->getConceptReference()->print(Out, Policy);
}

::testing::AssertionResult
PrintedConceptMatches(StringRef Code, const std::vector<std::string> &Args,
                      const StatementMatcher &NodeMatch,
                      StringRef ExpectedPrinted) {
  return PrintedNodeMatches<ConceptSpecializationExpr>(
      Code, Args, NodeMatch, ExpectedPrinted, "", PrintConceptReference);
}
const internal::VariadicDynCastAllOfMatcher<Stmt, ConceptSpecializationExpr>
    conceptSpecializationExpr;
} // unnamed namespace

TEST(ConceptPrinter, ConceptReference) {
  std::string Code = R"cpp(
    template <typename, typename> concept D = true;
    template<typename T, typename U>
    requires D<T, U>
    void g(T);
  )cpp";
  auto Matcher = conceptSpecializationExpr().bind("id");

  ASSERT_TRUE(PrintedConceptMatches(Code, {"-std=c++20"}, Matcher, "D<T, U>"));
}
