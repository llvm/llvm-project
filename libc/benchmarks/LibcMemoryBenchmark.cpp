//===-- Benchmark memory specific tools -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcMemoryBenchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

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
  for (size_t I = MaxSizeValue + 1; I < BufferSize; I += MaxSizeValue)
    MismatchIndices.push_back(I);
  if (MismatchIndices.empty())
    report_fatal_error("Unable to generate mismatch");
  MismatchIndexSelector =
      std::uniform_int_distribution<size_t>(0, MismatchIndices.size() - 1);
}

} // namespace libc_benchmarks
} // namespace llvm
