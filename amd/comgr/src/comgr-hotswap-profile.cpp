//===- comgr-hotswap-profile.cpp - HotSwap rewrite profiler --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Out-of-line definitions for the HotSwap rewrite profiler declared in
/// comgr-hotswap-internal.h, kept here rather than inline so the merge/flush
/// logic is not recompiled in every TU that includes the header.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

namespace COMGR {
namespace hotswap {

HotswapProfile::Scope::Scope(HotswapProfile *Profile, HotswapMetric Metric)
    : Profile(Profile), Metric(Metric), StartNs(Profile ? profNowNs() : 0) {}

void HotswapProfile::Scope::finish() {
  if (!Profile)
    return;
  Profile->add(Metric, profNowNs() - StartNs, Patches);
  Profile = nullptr;
}

HotswapProfile::Scope HotswapProfile::time(HotswapMetric Metric) {
  return Scope(Enabled ? this : nullptr, Metric);
}

void HotswapProfile::count(HotswapMetric Metric, uint64_t N) {
  if (Enabled)
    Samples[static_cast<size_t>(Metric)].Calls += N;
}

void HotswapProfile::add(HotswapMetric Metric, uint64_t Nanos,
                         uint64_t Patches) {
  if (!Enabled)
    return;
  HotswapSample &S = Samples[static_cast<size_t>(Metric)];
  S.Nanos += Nanos;
  S.Calls += 1;
  S.Patches += Patches;
  S.MinNanos = std::min(S.MinNanos, Nanos);
  S.MaxNanos = std::max(S.MaxNanos, Nanos);
}

const HotswapSample &HotswapProfile::sample(HotswapMetric Metric) const {
  return Samples[static_cast<size_t>(Metric)];
}

llvm::SmallVector<COMGR::TimeStatistics::PerfStatRecord, HotswapMetricCount>
HotswapProfile::buildRecords(llvm::SmallVectorImpl<std::string> &Names) {
  uint64_t PhaseSum = 0;
  for (size_t I = 0; I < HotswapMetricCount; ++I)
    if (hotswapMetricInfo[I].PartitionsTotal)
      PhaseSum += Samples[I].Nanos;
  HotswapSample &Total =
      Samples[static_cast<size_t>(HotswapMetric::RewriteTotal)];
  HotswapSample &Unacc =
      Samples[static_cast<size_t>(HotswapMetric::Unaccounted)];
  Unacc.Nanos = Total.Nanos > PhaseSum ? Total.Nanos - PhaseSum : 0;
  Unacc.Calls = Total.Calls;
  // One unaccounted sample per rewrite: min == max == residual so the merged
  // min/max across rewrites stays correct.
  Unacc.MinNanos = Unacc.MaxNanos = Unacc.Nanos;

  const double UnitsPerNs = env::getGranularityUnitsPerSecond() / 1.0e9;
  // Append row names to the caller's storage (keeps them stable), then point
  // the records' StringRefs at them.
  const size_t NameBase = Names.size();
  llvm::SmallVector<size_t, HotswapMetricCount> Rows;
  for (size_t I = 0; I < HotswapMetricCount; ++I) {
    const HotswapSample &S = Samples[I];
    const HotswapMetric M = static_cast<HotswapMetric>(I);
    if (!S.Calls && !S.Nanos && !S.Patches && M != HotswapMetric::Unaccounted)
      continue;
    const HotswapMetricInfo &Info = hotswapMetricInfo[I];
    if (Info.Parent != HotswapMetric::Count)
      Names.push_back(
          std::string(
              hotswapMetricInfo[static_cast<size_t>(Info.Parent)].Label) +
          "/" + Info.Label);
    else
      Names.push_back(std::string(Info.Label));
    Rows.push_back(I);
  }

  llvm::SmallVector<COMGR::TimeStatistics::PerfStatRecord, HotswapMetricCount>
      Records;
  Records.reserve(Rows.size());
  for (size_t K = 0; K < Rows.size(); ++K) {
    const HotswapSample &S = Samples[Rows[K]];
    COMGR::TimeStatistics::PerfStatRecord R;
    R.Name = Names[NameBase + K];
    R.TimeTaken = static_cast<double>(S.Nanos) * UnitsPerNs;
    R.Calls = S.Calls;
    R.Patches = S.Patches;
    R.MinTime = S.MinNanos == std::numeric_limits<uint64_t>::max()
                    ? 0.0
                    : static_cast<double>(S.MinNanos) * UnitsPerNs;
    R.MaxTime = static_cast<double>(S.MaxNanos) * UnitsPerNs;
    Records.push_back(R);
  }
  return Records;
}

void HotswapProfile::flush() {
  llvm::SmallVector<std::string, HotswapMetricCount> Names;
  COMGR::TimeStatistics::mergeStats(buildRecords(Names));
}

} // namespace hotswap
} // namespace COMGR
