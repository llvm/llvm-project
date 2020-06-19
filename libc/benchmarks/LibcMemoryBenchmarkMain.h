//===-- BenchmarkRunner interface -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_MAIN_H
#define LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_MAIN_H

#include "LibcBenchmark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace libc_benchmarks {

// Each memory function benchmark implements this interface.
// It is used by the main function to run all benchmarks in a uniform manner.
class BenchmarkRunner {
public:
  virtual ~BenchmarkRunner() {}

  // Returns a list of all available functions to test.
  virtual ArrayRef<StringRef> getFunctionNames() const = 0;

  // Performs the benchmarking for a particular FunctionName and Size.
  virtual BenchmarkResult benchmark(const BenchmarkOptions &Options,
                                    StringRef FunctionName, size_t Size) = 0;
};

} // namespace libc_benchmarks
} // namespace llvm

#endif // LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_MAIN_H
