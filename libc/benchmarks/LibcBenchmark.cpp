//===-- Benchmark function -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcBenchmark.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Host.h"

namespace llvm {
namespace libc_benchmarks {

void checkRequirements() {
  const auto &CpuInfo = benchmark::CPUInfo::Get();
  if (CpuInfo.scaling == benchmark::CPUInfo::ENABLED)
    report_fatal_error(
        "CPU scaling is enabled, the benchmark real time measurements may be "
        "noisy and will incur extra overhead.");
}

HostState HostState::get() {
  const auto &CpuInfo = benchmark::CPUInfo::Get();
  HostState H;
  H.CpuFrequency = CpuInfo.cycles_per_second;
  H.CpuName = llvm::sys::getHostCPUName().str();
  for (const auto &BenchmarkCacheInfo : CpuInfo.caches) {
    CacheInfo CI;
    CI.Type = BenchmarkCacheInfo.type;
    CI.Level = BenchmarkCacheInfo.level;
    CI.Size = BenchmarkCacheInfo.size;
    CI.NumSharing = BenchmarkCacheInfo.num_sharing;
    H.Caches.push_back(std::move(CI));
  }
  return H;
}

} // namespace libc_benchmarks
} // namespace llvm
