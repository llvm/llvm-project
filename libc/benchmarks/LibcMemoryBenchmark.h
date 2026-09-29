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
#include "LibcFunctionPrototypes.h"
#include "MemorySizeDistributions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace llvm {
namespace libc_benchmarks {

//--------------
// Configuration
//--------------

struct StudyConfiguration {
  // One of 'memcpy', 'memset', 'memcmp'.
  // The underlying implementation is always the llvm libc one.
  // e.g. 'memcpy' will test 'LIBC_NAMESPACE::memcpy'
  std::string function;

  // The number of trials to run for this benchmark.
  // If in SweepMode, each individual sizes are measured 'NumTrials' time.
  // i.e 'NumTrials' measurements for 0, 'NumTrials' measurements for 1 ...
  uint32_t num_trials = 1;

  // Toggles between Sweep Mode and Distribution Mode (default).
  // See 'SweepModeMaxSize' and 'SizeDistributionName' below.
  bool is_sweep_mode = false;

  // Maximum size to use when measuring a ramp of size values (SweepMode).
  // The benchmark measures all sizes from 0 to SweepModeMaxSize.
  // Note: in sweep mode the same size is sampled several times in a row this
  // will allow the processor to learn it and optimize the branching pattern.
  // The resulting measurement is likely to be idealized.
  uint32_t sweep_mode_max_size = 0; // inclusive

  // The name of the distribution to be used to randomize the size parameter.
  // This is used when SweepMode is false (default).
  std::string size_distribution_name;

  // This parameter allows to control how the buffers are accessed during
  // benchmark:
  // None : Use a fixed address that is at least cache line aligned,
  //    1 : Use random address,
  //   >1 : Use random address aligned to value.
  MaybeAlign access_alignment = std::nullopt;

  // When Function == 'memcmp', this is the buffers mismatch position.
  //  0 : Buffers always compare equal,
  // >0 : Buffers compare different at byte N-1.
  uint32_t memcmp_mismatch_at = 0;
};

struct Runtime {
  // Details about the Host (cpu name, cpu frequency, cache hierarchy).
  HostState host;

  // The framework will populate this value so all data accessed during the
  // benchmark will stay in L1 data cache. This includes bookkeeping data.
  uint32_t buffer_size = 0;

  // This is the number of distinct parameters used in a single batch.
  // The framework always tests a batch of randomized parameter to prevent the
  // cpu from learning branching patterns.
  uint32_t batch_parameter_count = 0;

  // The benchmark options that were used to perform the measurement.
  // This is decided by the framework.
  llvm::libc_benchmarks::BenchmarkOptions benchmark_options;
};

//--------
// Results
//--------

// The root object containing all the data (configuration and measurements).
struct Study {
  std::string study_name;
  llvm::libc_benchmarks::Runtime runtime;
  StudyConfiguration configuration;
  std::vector<Duration> measurements;
};

//------
// Utils
//------

// Provides an aligned, dynamically allocated buffer.
class AlignedBuffer {
  char *const Buffer = nullptr;
  size_t Size = 0;

public:
  static constexpr size_t Alignment = 512;

  explicit AlignedBuffer(size_t Size)
      : Buffer(static_cast<char *>(
            aligned_alloc(Alignment, alignTo(Size, Alignment)))),
        Size(Size) {}
  ~AlignedBuffer() { free(Buffer); }

  inline char *operator+(size_t Index) { return Buffer + Index; }
  inline const char *operator+(size_t Index) const { return Buffer + Index; }
  inline char &operator[](size_t Index) { return Buffer[Index]; }
  inline const char &operator[](size_t Index) const { return Buffer[Index]; }
  inline char *begin() { return Buffer; }
  inline char *end() { return Buffer + Size; }
};

// Helper to generate random buffer offsets that satisfy the configuration
// constraints.
class OffsetDistribution {
  std::uniform_int_distribution<uint32_t> Distribution;
  uint32_t Factor;

public:
  explicit OffsetDistribution(size_t BufferSize, size_t MaxSizeValue,
                              MaybeAlign AccessAlignment);

  template <class Generator> uint32_t operator()(Generator &G) {
    return Distribution(G) * Factor;
  }
};

#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
using MismatchIndicesType = llvm::SmallVector<uint32_t, 16>;
using MismatchIndicesImplType = llvm::SmallVectorImpl<uint32_t>;
#else
using MismatchIndicesType = std::vector<uint32_t>;
using MismatchIndicesImplType = std::vector<uint32_t>;
#endif

// Helper to generate random buffer offsets that satisfy the configuration
// constraints. It is specifically designed to benchmark `memcmp` functions
// where we may want the Nth byte to differ.
class MismatchOffsetDistribution {
  std::uniform_int_distribution<size_t> MismatchIndexSelector;
  MismatchIndicesType MismatchIndices;
  const uint32_t MismatchAt;

public:
  explicit MismatchOffsetDistribution(size_t BufferSize, size_t MaxSizeValue,
                                      size_t MismatchAt);

  explicit operator bool() const { return !MismatchIndices.empty(); }

  const MismatchIndicesImplType &getMismatchIndices() const {
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

/// This structure holds a vector of ParameterType.
/// It makes sure that BufferCount x BufferSize Bytes and the vector of
/// ParameterType can all fit in the L1 cache.
struct ParameterBatch {
  struct ParameterType {
    unsigned offset_bytes : 16; // max : 16 KiB - 1
    unsigned size_bytes : 16;   // max : 16 KiB - 1
  };

  ParameterBatch(size_t BufferCount);

  /// Verifies that memory accessed through this parameter is valid.
  void check_valid(const ParameterType &) const;

  /// Computes the number of bytes processed during within this batch.
  size_t get_batch_bytes() const;

  const size_t buffer_size;
  const size_t batch_size;
  std::vector<ParameterType> parameters;
};

/// Provides source and destination buffers for the Copy operation as well as
/// the associated size distributions.
struct CopySetup : public ParameterBatch {
  CopySetup();

  inline static const ArrayRef<MemorySizeDistribution> get_distributions() {
    return getMemcpySizeDistributions();
  }

  inline void *call(ParameterType parameter, MemcpyFunction memcpy_func) {
    return memcpy_func(dst_buffer + parameter.offset_bytes,
                       src_buffer + parameter.offset_bytes,
                       parameter.size_bytes);
  }

private:
  AlignedBuffer src_buffer;
  AlignedBuffer dst_buffer;
};

/// Provides source and destination buffers for the Move operation as well as
/// the associated size distributions.
struct MoveSetup : public ParameterBatch {
  MoveSetup();

  inline static const ArrayRef<MemorySizeDistribution> get_distributions() {
    return getMemmoveSizeDistributions();
  }

  inline void *call(ParameterType parameter, MemmoveFunction memmove_func) {
    return memmove_func(buffer + ParameterBatch::buffer_size / 3,
                        buffer + parameter.offset_bytes, parameter.size_bytes);
  }

private:
  AlignedBuffer buffer;
};

/// Provides destination buffer for the Set operation as well as the associated
/// size distributions.
struct SetSetup : public ParameterBatch {
  SetSetup();

  inline static const ArrayRef<MemorySizeDistribution> get_distributions() {
    return getMemsetSizeDistributions();
  }

  inline void *call(ParameterType parameter, MemsetFunction memset_func) {
    return memset_func(dst_buffer + parameter.offset_bytes,
                       parameter.offset_bytes % 0xFF, parameter.size_bytes);
  }

  inline void *call(ParameterType parameter, BzeroFunction bzero_func) {
    bzero_func(dst_buffer + parameter.offset_bytes, parameter.size_bytes);
    return dst_buffer.begin();
  }

private:
  AlignedBuffer dst_buffer;
};

/// Provides left and right buffers for the Comparison operation as well as the
/// associated size distributions.
struct ComparisonSetup : public ParameterBatch {
  ComparisonSetup();

  inline static const ArrayRef<MemorySizeDistribution> get_distributions() {
    return getMemcmpSizeDistributions();
  }

  inline int call(ParameterType parameter,
                  MemcmpOrBcmpFunction memcmp_or_bcmp_func) {
    return memcmp_or_bcmp_func(lhs_buffer + parameter.offset_bytes,
                               rhs_buffer + parameter.offset_bytes,
                               parameter.size_bytes);
  }

private:
  AlignedBuffer lhs_buffer;
  AlignedBuffer rhs_buffer;
};

} // namespace libc_benchmarks
} // namespace llvm

#endif // LLVM_LIBC_UTILS_BENCHMARK_MEMORY_BENCHMARK_H
