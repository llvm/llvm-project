//===-- Benchmark memory specific tools -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file complements the `benchmark` header with memory specific tools and
// benchmarking facilities.

#ifndef LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_H
#define LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_H

#include "LibcBenchmark.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"
#include <cstdint>
#include <random>

namespace llvm {
namespace libc_benchmarks {

//--------------
// Configuration
//--------------

// Specifies a range of sizes to explore.
struct SizeRange {
  uint32_t From = 0;  // Inclusive
  uint32_t To = 1024; // Inclusive
  uint32_t Step = 1;
};

// An object to define how to test a memory function.
struct StudyConfiguration {
  // The number of run for the study.
  uint32_t Runs = 1;

  // The size of the buffers (1 buffer for memset but 2 for memcpy or memcmp).
  // When testing small sizes, it's important to keep the total allocated
  // size under the size of the L1 cache (usually 16 or 32KiB). The framework
  // will also use 2KiB of additional L1 memory to store the function
  // parameters.
  uint32_t BufferSize = 8192;

  // The range of sizes to exercise.
  SizeRange Size;

  MaybeAlign AddressAlignment; //  Unset : Use start of buffer which is at
                               //         least cache line aligned)
                               //     1 : Use random address,
                               //    >1 : Use random address aligned to value.

  // The value to use for memset.
  uint8_t MemsetValue = 0;

  // The mismatch position for memcmp.
  uint32_t MemcmpMismatchAt = 0; //  0 : Buffer compare equal,
                                 // >0 : Buffer compare different at byte N-1.
};

//--------
// Results
//--------

// The time to run one iteration of the function under test for the specified
// Size.
struct Measurement {
  uint32_t Size = 0;
  Duration Runtime = {};
};

// The measurements for a specific function.
struct FunctionMeasurements {
  std::string Name;
  std::vector<Measurement> Measurements;
};

// The root object containing all the data (configuration and measurements).
struct Study {
  HostState Host;
  BenchmarkOptions Options;
  StudyConfiguration Configuration;
  SmallVector<FunctionMeasurements, 4> Functions;
};

// Provides an aligned, dynamically allocated buffer.
class AlignedBuffer {
  char *const Buffer = nullptr;
  size_t Size = 0;

public:
  static constexpr size_t Alignment = 1024;

  explicit AlignedBuffer(size_t Size)
      : Buffer(static_cast<char *>(aligned_alloc(1024, Size))), Size(Size) {}
  ~AlignedBuffer() { free(Buffer); }

  inline char *operator+(size_t Index) { return Buffer + Index; }
  inline const char *operator+(size_t Index) const { return Buffer + Index; }
  inline char &operator[](size_t Index) { return Buffer[Index]; }
  inline const char &operator[](size_t Index) const { return Buffer[Index]; }
  inline char *begin() { return Buffer; }
  inline char *end() { return Buffer + Size; }
};

// Implements the ParameterProvider abstraction needed by the `benchmark`
// function. This implementation makes sure that all parameters will fit into
// `StorageSize` bytes. The total memory accessed during benchmark should be
// less than the data L1 cache, that is the storage for the ParameterProvider
// and the memory buffers.
template <typename Context, size_t StorageSize = 8 * 1024>
class SmallParameterProvider {
  using ParameterType = typename Context::ParameterType;
  ByteConstrainedArray<ParameterType, StorageSize> Parameters;
  size_t LastIterations;
  Context &Ctx;

public:
  explicit SmallParameterProvider(Context &C) : Ctx(C) {}
  SmallParameterProvider(const SmallParameterProvider &) = delete;
  SmallParameterProvider &operator=(const SmallParameterProvider &) = delete;

  // Useful to compute the histogram of the size parameter.
  CircularArrayRef<ParameterType> getLastBatch() const {
    return cycle(Parameters, LastIterations);
  }

  // Implements the interface needed by the `benchmark` function.
  CircularArrayRef<ParameterType> generateBatch(size_t Iterations) {
    LastIterations = Iterations;
    Ctx.Randomize(Parameters);
    return getLastBatch();
  }
};

// Helper to generate random buffer offsets that satisfy the configuration
// constraints.
class OffsetDistribution {
  std::uniform_int_distribution<uint32_t> Distribution;
  uint32_t Factor;

public:
  explicit OffsetDistribution(const StudyConfiguration &Conf);

  template <class Generator> uint32_t operator()(Generator &G) {
    return Distribution(G) * Factor;
  }
};

// Helper to generate random buffer offsets that satisfy the configuration
// constraints. It is specifically designed to benchmark `memcmp` functions
// where we may want the Nth byte to differ.
class MismatchOffsetDistribution {
  std::uniform_int_distribution<size_t> MismatchIndexSelector;
  llvm::SmallVector<uint32_t, 16> MismatchIndices;
  const uint32_t MismatchAt;

public:
  explicit MismatchOffsetDistribution(const StudyConfiguration &Conf);

  explicit operator bool() const { return !MismatchIndices.empty(); }

  const llvm::SmallVectorImpl<uint32_t> &getMismatchIndices() const {
    return MismatchIndices;
  }

  template <class Generator> uint32_t operator()(Generator &G, uint32_t Size) {
    const uint32_t MismatchIndex = MismatchIndices[MismatchIndexSelector(G)];
    // We need to position the offset so that a mismatch occurs at MismatchAt.
    if (Size >= MismatchAt)
      return MismatchIndex - MismatchAt;
    // Size is too small to trigger the mismatch.
    return MismatchIndex - Size - 1;
  }
};

} // namespace libc_benchmarks
} // namespace llvm

#endif // LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_H
