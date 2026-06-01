//===-- Benchmark function -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcBenchmark.h"
#include "llvm/ADT/StringRef.h"
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
#include "llvm/TargetParser/Host.h"
#endif

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
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
  H.CpuName = llvm::sys::getHostCPUName().str();
#else
  H.CpuName = "";
#endif
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

#ifndef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
#include "llvm/ADT/Twine.h"
#include <cstdlib>
#include <iostream>

namespace llvm {
void report_fatal_error(const char *reason, bool gen_crash_diag) {
  std::cerr << "Fatal error: " << reason << std::endl;
  std::abort();
}
void report_fatal_error(StringRef reason, bool gen_crash_diag) {
  std::cerr << "Fatal error: " << reason.str() << std::endl;
  std::abort();
}
void report_fatal_error(const Twine &reason, bool gen_crash_diag) {
  std::cerr << "Fatal error: (twine reason)" << std::endl;
  std::abort();
}
} // namespace llvm
#endif
