//===-- Benchmark memory specific tools -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcMemoryBenchmark.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#endif
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <optional>

#if !defined(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT) && __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace llvm {
namespace libc_benchmarks {

// Returns a distribution that samples the buffer to satisfy the required
// alignment.
// When alignment is set, the distribution is scaled down by `Factor` and scaled
// up again by the same amount during sampling.
static std::uniform_int_distribution<uint32_t>
getOffsetDistribution(size_t BufferSize, size_t MaxSizeValue,
                      MaybeAlign AccessAlignment) {
  if (AccessAlignment && *AccessAlignment > AlignedBuffer::Alignment)
    report_fatal_error(
        "AccessAlignment must be less or equal to AlignedBuffer::Alignment");
  if (!AccessAlignment)
    return std::uniform_int_distribution<uint32_t>(0, 0); // Always 0.
  // If we test up to Size bytes, the returned offset must stay under
  // BuffersSize - Size.
  int64_t MaxOffset = BufferSize;
  MaxOffset -= MaxSizeValue;
  MaxOffset -= 1;
  if (MaxOffset < 0)
    report_fatal_error(
        "BufferSize too small to exercise specified Size configuration");
  MaxOffset /= AccessAlignment->value();
  return std::uniform_int_distribution<uint32_t>(0, MaxOffset);
}

OffsetDistribution::OffsetDistribution(size_t BufferSize, size_t MaxSizeValue,
                                       MaybeAlign AccessAlignment)
    : Distribution(
          getOffsetDistribution(BufferSize, MaxSizeValue, AccessAlignment)),
      Factor(AccessAlignment.valueOrOne().value()) {}

// Precomputes offset where to insert mismatches between the two buffers.
MismatchOffsetDistribution::MismatchOffsetDistribution(size_t BufferSize,
                                                       size_t MaxSizeValue,
                                                       size_t MismatchAt)
    : MismatchAt(MismatchAt) {
  if (MismatchAt <= 1)
    return;
  for (size_t i = MaxSizeValue + 1; i < BufferSize; i += MaxSizeValue)
    MismatchIndices.push_back(i);
  if (MismatchIndices.empty())
    report_fatal_error("Unable to generate mismatch");
  MismatchIndexSelector =
      std::uniform_int_distribution<size_t>(0, MismatchIndices.size() - 1);
}

static size_t getL1DataCacheSize() {
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
  const std::vector<CacheInfo> &cache_infos = HostState::get().caches;
  const auto is_l1_data_cache = [](const CacheInfo &ci) {
    return ci.type == "Data" && ci.level == 1;
  };
  const auto cache_it = find_if(cache_infos, is_l1_data_cache);
  if (cache_it != cache_infos.end())
    return cache_it->size;
#elif defined(_SC_LEVEL1_DCACHE_SIZE)
  long res = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  if (res > 0)
    return static_cast<size_t>(res);
#endif
  report_fatal_error("Unable to read L1 Cache Data Size");
}

static constexpr int64_t KiB = 1024;
static constexpr int64_t ParameterStorageBytes = 4 * KiB;
static constexpr int64_t L1LeftAsideBytes = 1 * KiB;

static size_t getAvailableBufferSize() {
  return getL1DataCacheSize() - L1LeftAsideBytes - ParameterStorageBytes;
}

ParameterBatch::ParameterBatch(size_t BufferCount)
    : buffer_size(getAvailableBufferSize() / BufferCount),
      batch_size(ParameterStorageBytes / sizeof(ParameterType)),
      parameters(batch_size) {
  if (buffer_size <= 0 || batch_size < 100)
    report_fatal_error("Not enough L1 cache");
  const size_t parameter_bytes = parameters.size() * sizeof(ParameterType);
  const size_t buffer_bytes = buffer_size * BufferCount;
  if (parameter_bytes + buffer_bytes + L1LeftAsideBytes > getL1DataCacheSize())
    report_fatal_error(
        "We're splitting a buffer of the size of the L1 cache between a data "
        "buffer and a benchmark parameters buffer, so by construction the "
        "total should not exceed the size of the L1 cache");
}

size_t ParameterBatch::get_batch_bytes() const {
  size_t batch_bytes = 0;
  for (auto &p : parameters)
    batch_bytes += p.size_bytes;
  return batch_bytes;
}

void ParameterBatch::check_valid(const ParameterType &p) const {
  if (p.offset_bytes + p.size_bytes >= buffer_size) {
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
    report_fatal_error(
        llvm::Twine("Call would result in buffer overflow: Offset=")
            .concat(llvm::Twine(p.offset_bytes))
            .concat(", Size=")
            .concat(llvm::Twine(p.size_bytes))
            .concat(", BufferSize=")
            .concat(llvm::Twine(buffer_size)));
#else
    std::string message = "Call would result in buffer overflow: Offset=" +
                          std::to_string(p.offset_bytes) +
                          ", Size=" + std::to_string(p.size_bytes) +
                          ", BufferSize=" + std::to_string(buffer_size);
    report_fatal_error(message.c_str());
#endif
  }
}

CopySetup::CopySetup()
    : ParameterBatch(2), src_buffer(ParameterBatch::buffer_size),
      dst_buffer(ParameterBatch::buffer_size) {}

MoveSetup::MoveSetup()
    : ParameterBatch(3), buffer(ParameterBatch::buffer_size * 3) {}

ComparisonSetup::ComparisonSetup()
    : ParameterBatch(2), lhs_buffer(ParameterBatch::buffer_size),
      rhs_buffer(ParameterBatch::buffer_size) {
  // The memcmp buffers always compare equal.
  memset(lhs_buffer.begin(), 0xF, buffer_size);
  memset(rhs_buffer.begin(), 0xF, buffer_size);
}

SetSetup::SetSetup()
    : ParameterBatch(1), dst_buffer(ParameterBatch::buffer_size) {}

} // namespace libc_benchmarks
} // namespace llvm
