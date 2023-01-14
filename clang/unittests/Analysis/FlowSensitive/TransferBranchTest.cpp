//===- unittests/Analysis/FlowSensitive/SignAnalysisTest.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a test for the transferBranch function of the
//  TypeErasedDataflowAnalysis.
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang::dataflow::test {
namespace {

using namespace ast_matchers;

struct TestLattice {
  std::optional<bool> Branch;
  static TestLattice bottom() { return {}; }

  // Does not matter for this test, but we must provide some definition of join.
  LatticeJoinEffect join(const TestLattice &Other) {
    return LatticeJoinEffect::Unchanged;
  }
  friend bool operator==(const TestLattice &Lhs, const TestLattice &Rhs) {
    return Lhs.Branch == Rhs.Branch;
  }
};

class TestPropagationAnalysis
    : public DataflowAnalysis<TestPropagationAnalysis, TestLattice> {
public:
  explicit TestPropagationAnalysis(ASTContext &Context)
      : DataflowAnalysis<TestPropagationAnalysis, TestLattice>(Context) {}
  static TestLattice initialElement() { return TestLattice::bottom(); }
  void transfer(const CFGElement *, TestLattice &, Environment &) {}
  void transferBranch(bool Branch, const Stmt *S, TestLattice &L,
                      Environment &Env) {
    L.Branch = Branch;
  }
};

using ::testing::UnorderedElementsAre;

template <typename Matcher>
void runDataflow(llvm::StringRef Code, Matcher VerifyResults,
                 LangStandard::Kind Std = LangStandard::lang_cxx17,
                 llvm::StringRef TargetFun = "fun") {
  using ast_matchers::hasName;
  ASSERT_THAT_ERROR(
      checkDataflow<TestPropagationAnalysis>(
          AnalysisInputs<TestPropagationAnalysis>(
              Code, hasName(TargetFun),
              [](ASTContext &C, Environment &) {
                return TestPropagationAnalysis(C);
              })
              .withASTBuildArgs(
                  {"-fsyntax-only", "-fno-delayed-template-parsing",
                   "-std=" +
                       std::string(LangStandard::getLangStandardForKind(Std)
                                       .getName())}),
          VerifyResults),
      llvm::Succeeded());
}

template <typename LatticeT>
const LatticeT &getLatticeAtAnnotation(
    const llvm::StringMap<DataflowAnalysisState<LatticeT>> &AnnotationStates,
    llvm::StringRef Annotation) {
  auto It = AnnotationStates.find(Annotation);
  assert(It != AnnotationStates.end());
  return It->getValue().Lattice;
}

TEST(TransferBranchTest, IfElse) {
  std::string Code = R"(
    void fun(int a) {
      if (a > 0) {
        (void)1;
        // [[p]]
      } else {
        (void)0;
        // [[q]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<TestLattice>> &Results,
         const AnalysisOutputs &) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q"));

        const TestLattice &LP = getLatticeAtAnnotation(Results, "p");
        EXPECT_THAT(LP.Branch, std::make_optional(true));

        const TestLattice &LQ = getLatticeAtAnnotation(Results, "q");
        EXPECT_THAT(LQ.Branch, std::make_optional(false));
      },
      LangStandard::lang_cxx17);
}

} // namespace
} // namespace clang::dataflow::test
