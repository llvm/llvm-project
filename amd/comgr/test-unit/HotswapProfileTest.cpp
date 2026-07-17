//===- HotswapProfileTest.cpp - Unit tests for HotSwap profiling ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the opt-in HotSwap rewrite profiler
/// (comgr-hotswap-internal.h). The profiler is always compiled and gated only
/// at run time (AMD_COMGR_TIME_STATISTICS), so these tests exercise both the
/// disabled and enabled per-rewrite sessions directly. See review on
/// ROCm/llvm-project#3364 and #3388.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "time-stat/time-stat.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>
#include <thread>
#include <vector>

using namespace COMGR::hotswap;

// A disabled session is inert: every hook is a no-op, nothing lands in samples.
TEST(HotswapProfile, DisabledSessionRecordsNothing) {
  HotswapProfile Profile(/*Enabled=*/false);
  EXPECT_FALSE(Profile.enabled());

  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::Decode);
    S.addPatches(7);
  }
  Profile.count(HotswapMetric::JumpShort, 4);
  Profile.add(HotswapMetric::Trampoline, 1000, 2);

  EXPECT_EQ(Profile.sample(HotswapMetric::Decode).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::JumpShort).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Nanos, 0u);
}

// add() folds a pre-measured interval as one call, tracking totals and min/max.
TEST(HotswapProfile, EnabledSessionAddAccumulates) {
  HotswapProfile Profile(/*Enabled=*/true);
  EXPECT_TRUE(Profile.enabled());

  Profile.add(HotswapMetric::Decode, 500, 2);
  Profile.add(HotswapMetric::Decode, 1500, 3);

  const HotswapSample &S = Profile.sample(HotswapMetric::Decode);
  EXPECT_EQ(S.Calls, 2u);
  EXPECT_EQ(S.Nanos, 2000u);
  EXPECT_EQ(S.Patches, 5u);
  EXPECT_EQ(S.MinNanos, 500u);
  EXPECT_EQ(S.MaxNanos, 1500u);
}

// count() bumps only the call count (jump-outcome rows carry no wall time).
TEST(HotswapProfile, CountOnlyRecordsCalls) {
  HotswapProfile Profile(/*Enabled=*/true);
  Profile.count(HotswapMetric::JumpShort);
  Profile.count(HotswapMetric::JumpShort, 2);

  const HotswapSample &S = Profile.sample(HotswapMetric::JumpShort);
  EXPECT_EQ(S.Calls, 3u);
  EXPECT_EQ(S.Nanos, 0u);
  EXPECT_EQ(S.Patches, 0u);
}

// The RAII Scope records exactly one call (with its patches) on destruction.
TEST(HotswapProfile, ScopeRecordsOnceOnDestruction) {
  HotswapProfile Profile(/*Enabled=*/true);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::TrampolineDs2Addr);
    S.addPatches(1);
  }
  const HotswapSample &S = Profile.sample(HotswapMetric::TrampolineDs2Addr);
  EXPECT_EQ(S.Calls, 1u);
  EXPECT_EQ(S.Patches, 1u);
}

// finish() is idempotent: explicit finish() then destruction records one call.
TEST(HotswapProfile, ScopeFinishIsIdempotent) {
  HotswapProfile Profile(/*Enabled=*/true);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::WmmaSplit);
    S.addPatches(2);
    S.finish();
    S.finish();
  }
  const HotswapSample &S = Profile.sample(HotswapMetric::WmmaSplit);
  EXPECT_EQ(S.Calls, 1u);
  EXPECT_EQ(S.Patches, 2u);
}

// A Scope from a disabled session never records, even when patches are added.
TEST(HotswapProfile, DisabledScopeRecordsNothing) {
  HotswapProfile Profile(/*Enabled=*/false);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::Trampoline);
    S.addPatches(9);
  }
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Calls, 0u);
}

// The label/parent/partition table must stay in lockstep with the enum.
TEST(HotswapProfile, MetricInfoTableWellFormed) {
  size_t PartitionCount = 0;
  for (size_t I = 0; I < HotswapMetricCount; ++I) {
    const HotswapMetricInfo &Info = hotswapMetricInfo[I];
    ASSERT_NE(Info.Label, nullptr);
    EXPECT_NE(Info.Label[0], '\0');
    if (Info.Parent != HotswapMetric::Count) {
      // A child's parent must itself be a top-level row.
      const size_t ParentIdx = static_cast<size_t>(Info.Parent);
      ASSERT_LT(ParentIdx, HotswapMetricCount);
      EXPECT_EQ(hotswapMetricInfo[ParentIdx].Parent, HotswapMetric::Count);
    }
    if (Info.PartitionsTotal)
      ++PartitionCount;
  }
  EXPECT_GT(PartitionCount, 0u);
  EXPECT_FALSE(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::RewriteTotal)]
          .PartitionsTotal);
  EXPECT_FALSE(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::Unaccounted)]
          .PartitionsTotal);
  EXPECT_STREQ(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::RewriteTotal)].Label,
      "phase:rewrite_total");
}

// flush()/buildRecords() derives phase:unaccounted = rewrite_total - sum of the
// partitioned phases, converts each sample's ns to the configured granularity
// unit, and encodes parent/child row names. Inspect the records against known
// samples (flush() merges this same set into TimeStatistics).
TEST(HotswapProfile, FlushDerivesUnaccountedAndConvertsUnits) {
  HotswapProfile Profile(/*Enabled=*/true);
  Profile.add(HotswapMetric::RewriteTotal, 10000, 0);
  Profile.add(HotswapMetric::Decode, 3000, 0);
  Profile.add(HotswapMetric::GrowElf, 2000, 0);
  // A strat child: exercises the parent/child row name and confirms a
  // non-partitioned row does not change the unaccounted residual.
  Profile.add(HotswapMetric::TrampolineDs2Addr, 1000, 4);

  llvm::SmallVector<std::string, HotswapMetricCount> Names;
  llvm::SmallVector<COMGR::TimeStatistics::PerfStatRecord, HotswapMetricCount>
      Records = Profile.buildRecords(Names);

  // buildRecords() writes the derived residual back into the samples:
  // 10000 - (3000 + 2000) = 5000. The strat child (1000) does not partition.
  EXPECT_EQ(Profile.sample(HotswapMetric::Unaccounted).Nanos, 5000u);

  llvm::StringMap<COMGR::TimeStatistics::PerfStatRecord> ByName;
  for (const COMGR::TimeStatistics::PerfStatRecord &R : Records)
    ByName[R.Name] = R;

  const double UnitsPerNs = COMGR::env::getGranularityUnitsPerSecond() / 1.0e9;
  ASSERT_TRUE(ByName.count("phase:rewrite_total"));
  EXPECT_DOUBLE_EQ(ByName["phase:rewrite_total"].TimeTaken,
                   10000.0 * UnitsPerNs);
  EXPECT_DOUBLE_EQ(ByName["phase:decode"].TimeTaken, 3000.0 * UnitsPerNs);
  EXPECT_DOUBLE_EQ(ByName["phase:grow_elf"].TimeTaken, 2000.0 * UnitsPerNs);

  ASSERT_TRUE(ByName.count("phase:unaccounted"));
  EXPECT_DOUBLE_EQ(ByName["phase:unaccounted"].TimeTaken, 5000.0 * UnitsPerNs);

  // Child rows carry the "parent/child" name and their patch counts.
  ASSERT_TRUE(ByName.count("strat:trampoline/ds_2addr"));
  EXPECT_EQ(ByName["strat:trampoline/ds_2addr"].Patches, 4u);
}

// PerfStats::mergeStats is the profiler's concurrency premise: many concurrent
// rewrites fold their local records into one shared, mutex-guarded map. Hammer
// it from several threads and confirm every record is accounted for (and that
// the shared map access is race-free under ASAN/TSAN).
TEST(TimeStatisticsMerge, ConcurrentMergeStatsIsRaceFree) {
  COMGR::TimeStatistics::PerfStats Stats;
  constexpr unsigned NumThreads = 8;
  constexpr unsigned MergesPerThread = 2000;

  auto Worker = [&Stats, MergesPerThread]() {
    for (unsigned I = 0; I < MergesPerThread; ++I) {
      COMGR::TimeStatistics::PerfStatRecord R;
      R.Name = "row";
      R.TimeTaken = 1.0;
      R.Calls = 1;
      R.Patches = 2;
      R.MinTime = 1.0;
      R.MaxTime = 1.0;
      Stats.mergeStats(R);
    }
  };

  std::vector<std::thread> Threads;
  for (unsigned T = 0; T < NumThreads; ++T)
    Threads.emplace_back(Worker);
  for (std::thread &T : Threads)
    T.join();

  const COMGR::TimeStatistics::ProfileData D = Stats.lookupForTest("row");
  const unsigned Total = NumThreads * MergesPerThread;
  EXPECT_EQ(D.Counter, static_cast<int>(Total));
  EXPECT_EQ(D.Patches, static_cast<uint64_t>(Total) * 2);
  EXPECT_DOUBLE_EQ(D.TimeTaken, static_cast<double>(Total));
  EXPECT_DOUBLE_EQ(D.MinTime, 1.0);
  EXPECT_DOUBLE_EQ(D.MaxTime, 1.0);
}
