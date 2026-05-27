//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf binning.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/flat_tlsf/binning.h"
#include "src/__support/flat_tlsf/common.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::max;
using LIBC_NAMESPACE::cpp::nullopt;
using LIBC_NAMESPACE::cpp::numeric_limits;
using LIBC_NAMESPACE::cpp::optional;
using LIBC_NAMESPACE::flat_tlsf::Binning;
using LIBC_NAMESPACE::flat_tlsf::CHUNK_UNIT;

namespace {

class LlvmLibcFlatTlsfBinningTest : public LIBC_NAMESPACE::testing::Test {
protected:
  template <typename F>
  size_t find_binning_boundary(uint32_t next_bin, size_t base, size_t end,
                               F &&size_to_bin) {
    while (base < end) {
      size_t mid = base + (end - base) / 2;
      if (size_to_bin(mid) >= next_bin)
        end = mid;
      else
        base = mid + 1;
    }
    return base;
  }

  template <typename F1, typename F2>
  void find_binning_boundaries(size_t start_from_size,
                               optional<uint32_t> stop_at_bin, F1 &&size_to_bin,
                               F2 &&bin_boundary_callback) {
    size_t prev_size = start_from_size;
    size_t size = start_from_size;
    size_t increment = 1;

    optional<uint32_t> prev_bin;

    while (true) {
      uint32_t bin = size_to_bin(size);

      if (!prev_bin.has_value() || prev_bin.value() != bin) {
        if (prev_bin.has_value())
          size = find_binning_boundary(prev_bin.value() + 1, prev_size, size,
                                       size_to_bin);

        bin_boundary_callback(bin, size);

        increment = max<size_t>((size - prev_size) / 4, 1);
        prev_size = size;

        prev_bin = optional<uint32_t>(bin);
      }

      if (size != numeric_limits<size_t>::max()) {
        if (size <= numeric_limits<size_t>::max() - increment)
          size += increment;
        else
          size = numeric_limits<size_t>::max();
      } else {
        break;
      }

      if (stop_at_bin.has_value() && stop_at_bin.value() == bin)
        break;
    }
  }

  template <typename F>
  void check_binning_properties(optional<uint32_t> stop_at_bin,
                                F &&size_to_bin) {
    optional<uint32_t> prev_bin;

    auto callback = [&](uint32_t bin, size_t size) {
      if (prev_bin.has_value())
        EXPECT_EQ(prev_bin.value() + 1, bin);
      prev_bin = optional<uint32_t>(bin);

      EXPECT_TRUE(
          !stop_at_bin.has_value() || bin <= stop_at_bin.value() ||
          (bin == numeric_limits<uint32_t>::max() && size < CHUNK_UNIT));
    };

    find_binning_boundaries(CHUNK_UNIT - 1, stop_at_bin, size_to_bin, callback);
  }
};

TEST_F(LlvmLibcFlatTlsfBinningTest, CheckFindBinningBoundary) {
  array<uint32_t, 12> size_to_bin = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 5};
  auto size_to_bin_fn = [&](size_t s) { return size_to_bin[s]; };

  EXPECT_EQ(find_binning_boundary(2, 1, 3, size_to_bin_fn), size_t{3});
  EXPECT_EQ(find_binning_boundary(2, 3, 5, size_to_bin_fn), size_t{4});
  EXPECT_EQ(find_binning_boundary(2, 2, 4, size_to_bin_fn), size_t{4});
  EXPECT_EQ(find_binning_boundary(2, 4, 6, size_to_bin_fn), size_t{4});
  EXPECT_EQ(find_binning_boundary(2, 5, 7, size_to_bin_fn), size_t{5});

  EXPECT_EQ(find_binning_boundary(2, 2, 11, size_to_bin_fn), size_t{4});
  EXPECT_EQ(find_binning_boundary(2, 0, 7, size_to_bin_fn), size_t{4});

  EXPECT_EQ(find_binning_boundary(4, 0, 11, size_to_bin_fn), size_t{11});
}

TEST_F(LlvmLibcFlatTlsfBinningTest, CheckFindBinningBoundaries) {
  array<uint32_t, 12> size_to_bin = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3};
  array<size_t, 4> boundary_sizes = {0, 1, 4, 10};

  size_t i = 0;
  auto verifier = [&](uint32_t bin, size_t size) {
    ASSERT_LT(i, boundary_sizes.size());
    EXPECT_EQ(size, boundary_sizes[i]);
    EXPECT_EQ(bin, size_to_bin[size]);
    i++;
  };

  auto size_to_bin_fn = [&](size_t s) { return size_to_bin[s]; };

  find_binning_boundaries(0, 3, size_to_bin_fn, verifier);
  EXPECT_EQ(i, size_t{4});
}

TEST_F(LlvmLibcFlatTlsfBinningTest,
       TestLinearExtentThenLinearlyDividedExponentialBinning) {
  auto size_to_bin_fn = [](size_t size) {
    return static_cast<uint32_t>(
        Binning::linear_extend_then_linearly_divided_expotential_binning<8, 4>(
            size));
  };
  check_binning_properties(nullopt, size_to_bin_fn);
}

} // namespace
