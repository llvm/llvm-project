//===- unittests/StaticAnalyzer/Z3CrosscheckOracleTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/Z3CrosscheckVisitor.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

using Z3Result = Z3CrosscheckVisitor::Z3Result;
using Z3Decision = Z3CrosscheckOracle::Z3Decision;

static constexpr Z3Decision AcceptReport = Z3Decision::AcceptReport;
static constexpr Z3Decision RejectReport = Z3Decision::RejectReport;
static constexpr Z3Decision RejectEQClass = Z3Decision::RejectEQClass;

static constexpr std::optional<bool> SAT = true;
static constexpr std::optional<bool> UNSAT = false;
static constexpr std::optional<bool> UNDEF = std::nullopt;

static unsigned operator""_ms(unsigned long long ms) { return ms; }
static unsigned operator""_step(unsigned long long rlimit) { return rlimit; }

static const AnalyzerOptions DefaultOpts = [] {
  AnalyzerOptions Config;
#define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(TYPE, NAME, CMDFLAG, DESC,        \
                                             SHALLOW_VAL, DEEP_VAL)            \
  ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEEP_VAL)
#define ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL)                \
  Config.NAME = DEFAULT_VAL;
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"

  // Remember to update the tests in this file when these values change.
  // Also update the doc comment of `interpretQueryResult`.
  assert(Config.Z3CrosscheckRLimitThreshold == 400'000);
  assert(Config.Z3CrosscheckTimeoutThreshold == 300_ms);
  // Usually, when the timeout/rlimit threshold is reached, Z3 only slightly
  // overshoots until it realizes that it overshoot and needs to back off.
  // Consequently, the measured timeout should be fairly close to the threshold.
  // Same reasoning applies to the rlimit too.
  return Config;
}();

namespace {

class Z3CrosscheckOracleTest : public testing::Test {
public:
  Z3Decision interpretQueryResult(const Z3Result &Result) {
    return Oracle.interpretQueryResult(Result);
  }

private:
  Z3CrosscheckOracle Oracle = Z3CrosscheckOracle(DefaultOpts);
};

TEST_F(Z3CrosscheckOracleTest, AcceptsFirstSAT) {
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 25_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, AcceptsSAT) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 25_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, SATWhenItGoesOverTime) {
  // Even if it times out, if it is SAT, we should accept it.
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 310_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, UNSATWhenItGoesOverTime) {
  ASSERT_EQ(RejectEQClass, interpretQueryResult({UNSAT, 310_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, RejectsTimeout) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectEQClass, interpretQueryResult({UNDEF, 310_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, RejectsUNSATs) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
}

// Testing cut heuristics:
// =======================

TEST_F(Z3CrosscheckOracleTest, RejectEQClassIfSpendsTooMuchTotalTime) {
  // Simulate long queries, that barely doesn't trigger the timeout.
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 290_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 290_ms, 1000_step}));
  ASSERT_EQ(RejectEQClass, interpretQueryResult({UNSAT, 290_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, SATWhenItSpendsTooMuchTotalTime) {
  // Simulate long queries, that barely doesn't trigger the timeout.
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 290_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 290_ms, 1000_step}));
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 290_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, RejectEQClassIfAttemptsManySmallQueries) {
  // Simulate quick, but many queries: 35 quick UNSAT queries.
  // 35*20ms = 700ms, which is equal to the 700ms threshold.
  for (int i = 0; i < 35; ++i) {
    ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 20_ms, 1000_step}));
  }
  // Do one more to trigger the heuristic.
  ASSERT_EQ(RejectEQClass, interpretQueryResult({UNSAT, 1_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, SATWhenIfAttemptsManySmallQueries) {
  // Simulate quick, but many queries: 35 quick UNSAT queries.
  // 35*20ms = 700ms, which is equal to the 700ms threshold.
  for (int i = 0; i < 35; ++i) {
    ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 20_ms, 1000_step}));
  }
  // Do one more to trigger the heuristic, but given this was SAT, we still
  // accept the query.
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 200_ms, 1000_step}));
}

TEST_F(Z3CrosscheckOracleTest, RejectEQClassIfExhaustsRLimit) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectEQClass, interpretQueryResult({UNDEF, 25_ms, 405'000_step}));
}

TEST_F(Z3CrosscheckOracleTest, SATWhenItExhaustsRLimit) {
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(RejectReport, interpretQueryResult({UNSAT, 25_ms, 1000_step}));
  ASSERT_EQ(AcceptReport, interpretQueryResult({SAT, 25_ms, 405'000_step}));
}

} // namespace
