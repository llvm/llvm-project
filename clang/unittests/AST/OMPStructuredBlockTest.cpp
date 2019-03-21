//===- unittests/AST/OMPStructuredBlockTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fine-grained tests for IsOMPStructuredBlock bit of Stmt.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

AST_MATCHER(Stmt, isOMPStructuredBlock) { return Node.isOMPStructuredBlock(); }

const ast_matchers::internal::VariadicDynCastAllOfMatcher<
    Stmt, OMPExecutableDirective>
    ompExecutableDirective;

const ast_matchers::internal::VariadicDynCastAllOfMatcher<
    OMPExecutableDirective, OMPTargetDirective>
    ompTargetDirective;

StatementMatcher OMPInnermostStructuredBlockMatcher() {
  return stmt(isOMPStructuredBlock(),
              unless(hasDescendant(stmt(isOMPStructuredBlock()))))
      .bind("id");
}

AST_MATCHER(OMPExecutableDirective, isStandaloneDirective) {
  return Node.isStandaloneDirective();
}

StatementMatcher OMPStandaloneDirectiveMatcher() {
  return stmt(ompExecutableDirective(isStandaloneDirective())).bind("id");
}

template <typename T>
::testing::AssertionResult
PrintedOMPStmtMatches(StringRef Code, const T &NodeMatch,
                      StringRef ExpectedPrinted,
                      PolicyAdjusterType PolicyAdjuster = None) {
  std::vector<std::string> Args = {
      "-fopenmp",
  };
  return PrintedStmtMatches(Code, Args, NodeMatch, ExpectedPrinted,
                            PolicyAdjuster);
}

static testing::AssertionResult NoMatches(StringRef Code,
                                          const StatementMatcher &StmtMatch) {
  PrintMatch Printer((PolicyAdjusterType()));
  MatchFinder Finder;
  Finder.addMatcher(StmtMatch, &Printer);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  if (!runToolOnCode(Factory->create(), Code))
    return testing::AssertionFailure()
           << "Parsing error in \"" << Code.str() << "\"";
  if (Printer.getNumFoundStmts() == 0)
    return testing::AssertionSuccess();
  return testing::AssertionFailure()
         << "Matcher should match only zero statements (found "
         << Printer.getNumFoundStmts() << ")";
}

} // unnamed namespace

TEST(OMPStructuredBlock, TestAtomic) {
  const char *Source =
      R"(
void test(int i) {
#pragma omp atomic
++i;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), "++i"));
}

TEST(OMPStructuredBlock, TestBarrier) {
  const char *Source =
      R"(
void test() {
#pragma omp barrier
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp barrier\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestCancel) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel
{
    #pragma omp cancel parallel
}
})";
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPInnermostStructuredBlockMatcher(), R"({
    #pragma omp cancel parallel
}
)"));
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp cancel parallel\n"));
}

TEST(OMPStructuredBlock, TestCancellationPoint) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel
{
#pragma omp cancellation point parallel
}
})";
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPInnermostStructuredBlockMatcher(), R"({
    #pragma omp cancellation point parallel
}
)"));
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                            "#pragma omp cancellation point parallel\n"));
}

TEST(OMPStructuredBlock, TestCritical) {
  const char *Source =
      R"(
void test() {
#pragma omp critical
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

//----------------------------------------------------------------------------//
// Loop tests
//----------------------------------------------------------------------------//

class OMPStructuredBlockLoop : public ::testing::TestWithParam<const char *> {};

TEST_P(OMPStructuredBlockLoop, TestDirective0) {
  const std::string Source =
      R"(
void test(int x) {
#pragma omp )" +
      std::string(GetParam()) + R"(
for (int i = 0; i < x; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST_P(OMPStructuredBlockLoop, TestDirective1) {
  const std::string Source =
      R"(
void test(int x, int y) {
#pragma omp )" +
      std::string(GetParam()) + R"(
for (int i = 0; i < x; i++)
for (int i = 0; i < y; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source,
                                    OMPInnermostStructuredBlockMatcher(),
                                    "for (int i = 0; i < y; i++)\n    ;\n"));
}

TEST_P(OMPStructuredBlockLoop, TestDirectiveCollapse1) {
  const std::string Source =
      R"(
void test(int x, int y) {
#pragma omp )" +
      std::string(GetParam()) + R"( collapse(1)
for (int i = 0; i < x; i++)
for (int i = 0; i < y; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source,
                                    OMPInnermostStructuredBlockMatcher(),
                                    "for (int i = 0; i < y; i++)\n    ;\n"));
}

TEST_P(OMPStructuredBlockLoop, TestDirectiveCollapse2) {
  const std::string Source =
      R"(
void test(int x, int y) {
#pragma omp )" +
      std::string(GetParam()) + R"( collapse(2)
for (int i = 0; i < x; i++)
for (int i = 0; i < y; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST_P(OMPStructuredBlockLoop, TestDirectiveCollapse22) {
  const std::string Source =
      R"(
void test(int x, int y, int z) {
#pragma omp )" +
      std::string(GetParam()) + R"( collapse(2)
for (int i = 0; i < x; i++)
for (int i = 0; i < y; i++)
for (int i = 0; i < z; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source,
                                    OMPInnermostStructuredBlockMatcher(),
                                    "for (int i = 0; i < z; i++)\n    ;\n"));
}

INSTANTIATE_TEST_CASE_P(
    OMPStructuredBlockLoopDirectives, OMPStructuredBlockLoop,
    ::testing::Values("simd", "for", "for simd", "parallel for",
                      "parallel for simd", "target parallel for", "taskloop",
                      "taskloop simd", "distribute", "distribute parallel for",
                      "distribute parallel for simd", "distribute simd",
                      "target parallel for simd", "target simd",
                      "target\n#pragma omp teams distribute",
                      "target\n#pragma omp teams distribute simd",
                      "target\n#pragma omp teams distribute parallel for simd",
                      "target\n#pragma omp teams distribute parallel for",
                      "target teams distribute",
                      "target teams distribute parallel for",
                      "target teams distribute parallel for simd",
                      "target teams distribute simd"), );

//----------------------------------------------------------------------------//
// End Loop tests
//----------------------------------------------------------------------------//

TEST(OMPStructuredBlock, TestFlush) {
  const char *Source =
      R"(
void test() {
#pragma omp flush
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp flush\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestMaster) {
  const char *Source =
      R"(
void test() {
#pragma omp master
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestOrdered0) {
  const char *Source =
      R"(
void test() {
#pragma omp ordered
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestOrdered1) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp for ordered
for (int i = 0; i < x; i++)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestOrdered2) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp for ordered(1)
for (int i = 0; i < x; i++) {
#pragma omp ordered depend(source)
}
})";
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPInnermostStructuredBlockMatcher(),
                            "{\n    #pragma omp ordered depend(source)\n}\n"));
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp ordered depend(source)\n"));
}

TEST(OMPStructuredBlock, DISABLED_TestParallelMaster0XFAIL) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel master
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, DISABLED_TestParallelMaster1XFAIL) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel master
{ ; }
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), "{\n    ;\n}\n"));
}

TEST(OMPStructuredBlock, TestParallelSections) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel sections
{ ; }
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), "{\n    ;\n}\n"));
}

TEST(OMPStructuredBlock, TestParallelDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp parallel
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<
    OMPExecutableDirective, OMPSectionsDirective>
    ompSectionsDirective;

const ast_matchers::internal::VariadicDynCastAllOfMatcher<
    OMPExecutableDirective, OMPSectionDirective>
    ompSectionDirective;

StatementMatcher OMPSectionsDirectiveMatcher() {
  return stmt(
             isOMPStructuredBlock(),
             hasAncestor(ompExecutableDirective(ompSectionsDirective())),
             unless(hasAncestor(ompExecutableDirective(ompSectionDirective()))))
      .bind("id");
}

TEST(OMPStructuredBlock, TestSectionDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp sections
{
#pragma omp section
;
}
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPSectionsDirectiveMatcher(),
                                    "{\n"
                                    "    #pragma omp section\n"
                                    "        ;\n"
                                    "}\n"));
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestSections) {
  const char *Source =
      R"(
void test() {
#pragma omp sections
{ ; }
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), "{\n    ;\n}\n"));
}

TEST(OMPStructuredBlock, TestSingleDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp single
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TesTargetDataDirective) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp target data map(x)
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TesTargetEnterDataDirective) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp target enter data map(to : x)
})";
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                            "#pragma omp target enter data map(to: x)\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TesTargetExitDataDirective) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp target exit data map(from : x)
})";
  ASSERT_TRUE(
      PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                            "#pragma omp target exit data map(from: x)\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestTargetParallelDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp target parallel
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestTargetTeams) {
  const char *Source =
      R"(
void test() {
#pragma omp target teams
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestTargetUpdateDirective) {
  const char *Source =
      R"(
void test(int x) {
#pragma omp target update to(x)
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp target update to(x)\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestTarget) {
  const char *Source =
      R"(
void test() {
#pragma omp target
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestTask) {
  const char *Source =
      R"(
void test() {
#pragma omp task
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestTaskgroup) {
  const char *Source =
      R"(
void test() {
#pragma omp taskgroup
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}

TEST(OMPStructuredBlock, TestTaskwaitDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp taskwait
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp taskwait\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestTaskyieldDirective) {
  const char *Source =
      R"(
void test() {
#pragma omp taskyield
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(Source, OMPStandaloneDirectiveMatcher(),
                                    "#pragma omp taskyield\n"));
  ASSERT_TRUE(NoMatches(Source, OMPInnermostStructuredBlockMatcher()));
}

TEST(OMPStructuredBlock, TestTeams) {
  const char *Source =
      R"(
void test() {
#pragma omp target
#pragma omp teams
;
})";
  ASSERT_TRUE(PrintedOMPStmtMatches(
      Source, OMPInnermostStructuredBlockMatcher(), ";\n"));
}
