//===- unittests/StaticAnalyzer/Z3CrosscheckOracleTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/Z3CrosscheckVisitor.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

using Z3Result = Z3CrosscheckVisitor::Z3Result;
using Z3Decision = Z3CrosscheckOracle::Z3Decision;

static constexpr Z3Decision AcceptReport = Z3Decision::AcceptReport;
static constexpr Z3Decision RejectReport = Z3Decision::RejectReport;

static constexpr std::optional<bool> SAT = true;
static constexpr std::optional<bool> UNSAT = false;
static constexpr std::optional<bool> UNDEF = std::nullopt;

namespace {

struct Z3CrosscheckOracleTest : public testing::Test {
  Z3Decision interpretQueryResult(const Z3Result &Result) const {
    return Z3CrosscheckOracle::interpretQueryResult(Result);
  }
};

TEST_F(Z3CrosscheckOracleTest, AcceptsFirstSAT) {
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT}));
}

TEST_F(Z3CrosscheckOracleTest, AcceptsSAT) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT}));
}

TEST_F(Z3CrosscheckOracleTest, AcceptsFirstTimeout) {
  ASSERT_EQ(AcceptReport, interpretQueryResult({UNDEF}));
}

TEST_F(Z3CrosscheckOracleTest, AcceptsTimeout) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(AcceptReport, interpretQueryResult({UNDEF}));
}

TEST_F(Z3CrosscheckOracleTest, RejectsUNSATs) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT}));
}

} // namespace
