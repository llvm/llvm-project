//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide Binning class for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BINNING_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BINNING_H

#include "hdr/stdint_proxy.h"
#include "src/__support/flat_tlsf/bit_utils.h"
#include "src/__support/flat_tlsf/bitfield.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

struct Binning {
  static_assert(sizeof(void *) == 4 || sizeof(void *) == 8,
                "Only 32-bit and 64-bit architectures are currently supported");

  static constexpr size_t BIN_COUNT = BitField::BITS - 1;

  /// A fast binning algorithm with relatively even coverage and configurable
  /// behavior.
  ///
  /// This is the default binning algorithm used due to having a good spread of
  /// bin intervals, being able to take advantage of many or few buckets well,
  /// and being very fast (only a handful of instructions with one branch).
  ///
  /// # Behavior by size
  /// - `0..=(CHUNK_UNIT*LIN_DIVS*LIN_EXT_MULTI)` : Bins sizes into
  /// one-bin-per-chunk-size
  /// - `(CHUNK_UNIT*LIN_DIVS*LIN_EXT_MULTI)..`   : Bins sizes by
  /// linearly-subdivided exponential levels.
  ///
  /// # Parameters
  /// - `LIN_DIVS`: the number of linear regions per power of two in the
  /// exponential region.
  ///     - The higher this is, the more buckets are needed but the binning is
  ///     more fine-grained.
  ///     - Must be a power of two.
  ///     - Typically 2 (few bins, subpar granularity), 4, or 8 (lots of bins,
  ///     good granularity).
  ///     - This is the parameter you want to figure out first for a given
  ///     number of bins.
  ///
  /// - `LIN_EXT_MULTI`: the linear region extent multiplier.
  ///     - Scales the extent of the linear region.
  ///     - Must be a power of two.
  ///     - Set this to 1 by default.
  ///     - If there are too many bins being used on excessively-high size
  ///     regions, this is useful for spending those bins on more buckets for
  ///     small sizes instead.
  ///
  /// # Deciding on the parameters
  /// `LIN_DIVS` has a much larger effect so tinker with that first while
  /// keeping `LIN_EXT_MULTI` low, and then increase `LIN_EXT_MULTI` if there is
  /// useless range at the top, given the number of bins you have.
  ///
  /// Having a range up to around 128MiB~2GiB is enough for most applications.
  /// But keep in mind the largest bucket size you'll ever make use of is the
  /// largest contiguous span of memory.
  ///
  /// The main effects on the allocator will be the heap efficiency and the
  /// performance.
  template <size_t LIN_DIVS, size_t LIN_EXT_MULTI>
  LIBC_INLINE static constexpr size_t
  linear_extend_then_linearly_divided_exponential_binning(size_t size) {
    static_assert(bit_utils::is_power_of_2(LIN_DIVS),
                  "LIN_DIVS must be a power of two");
    static_assert(bit_utils::is_power_of_2(LIN_EXT_MULTI),
                  "LIN_EXT_MULTI must be a power of two");

    size_t exponential_region = CHUNK_UNIT * LIN_DIVS * LIN_EXT_MULTI;

    // If the size is small enough, just divide by the chunk size.
    // This is a fast short-circuit that handles the case where the size is
    // smaller than `CHUNK_UNIT` and doesn't waste extra bins due to exponential
    // subdivisions being smaller than `CHUNK_UNIT` here.
    if (size <= exponential_region)
      return size >> bit_utils::ilog2(CHUNK_UNIT);

    // Let's say `exponential_region` is 256, the chunk unit is 32, LIN_DIVS is
    // 4
    //
    // Exponential level 0:  256 ;  (512 - 256)/LIN_DIVS = 256/LIN_DIVS = 64
    //  Subdiv 0: 256       ; bin 0 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 1: 256 +  64 ; bin 1 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 2: 256 + 128 ; bin 2 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 3: 256 + 192 ; bin 3 + LIN_DIVS * LIN_EXT_MULTI
    // Exponential level 1:  512 ;  512/LIN_DIVS = 128
    //  Subdiv 0: 512       ; bin 4 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 1: 512 + 128 ; bin 5 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 2: 512 + 256 ; bin 6 + LIN_DIVS * LIN_EXT_MULTI
    //  Subdiv 3: 512 + 384 ; bin 7 + LIN_DIVS * LIN_EXT_MULTI
    // Exponential level 2: 1024 ; 1024/LIN_DIVS = 256
    //  etc...
    //
    // Any size here is essentially broken up as follows:
    //
    // 00000000_1_01_010101010
    //               ^^^^^^^^^ dead bits; all of this is ignored, effectively
    //               rounding down these bits away
    //            ^^ linear division bits; LIN_DIVS.ilog2() bits long after the
    //            first set bit; tells us which linear subdivision we're in
    //          ^ first set bit; dictates size.ilog2(); tells us which
    //          "exponential level" this size is

    size_t size_ilog2 = bit_utils::ilog2(size);

    // Shift out the dead bits. This leaves the linear subdivision plus LIN_DIVS
    // (due to the always-set bit at the top)
    size_t linear_subdivision_plus_lin_divs =
        size >> (size_ilog2 - bit_utils::ilog2(LIN_DIVS));

    // Extract the exponential level above the `exponential_region` limit
    // add LIN_EXT_MULTI here along with the other constants, it will get
    // multiplied by LIN_DIVS next which gives us the exponential bins offset
    // subtract 1 along with the other constants, this is important later
    size_t unshifted_exponential_minus_one =
        size_ilog2 - bit_utils::ilog2(exponential_region) + LIN_EXT_MULTI - 1;

    // Multiply the exponential level by LIN_DIVS to shift it above the linear
    // division bits Multiply the LIN_EXT_MULTI by LIN_DIVS to add the offset
    // due to the linearly-spaced buckets Multiply (-1) to get (-LIN_DIVS)
    size_t exponential_plus_offset_minus_lin_divs =
        unshifted_exponential_minus_one << bit_utils::ilog2(LIN_DIVS);

    // This LIN_DIVS cancel out, yielding the expected exponential-region bin
    return exponential_plus_offset_minus_lin_divs +
           linear_subdivision_plus_lin_divs;
  }

  LIBC_INLINE constexpr static uint32_t size_to_bin(size_t size) {
    if constexpr (sizeof(void *) == 8)
      return static_cast<uint32_t>(
          linear_extend_then_linearly_divided_exponential_binning<8, 4>(size));
    else
      return static_cast<uint32_t>(
          linear_extend_then_linearly_divided_exponential_binning<4, 4>(size));
  }

  LIBC_INLINE constexpr static uint32_t size_to_bin_ceil(size_t size) {
    return size_to_bin(size - 1) + 1;
  }
};

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BINNING_H
