//===-- A self contained equivalent of std::bitset --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_BITSET_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_BITSET_H

#include <stddef.h> // For size_t.

namespace __llvm_libc::cpp {

template <size_t NumberOfBits> struct bitset {
  static_assert(NumberOfBits != 0,
                "Cannot create a __llvm_libc::cpp::bitset of size 0.");

  constexpr void set(size_t Index) {
    Data[Index / BITS_PER_UNIT] |= mask(Index);
  }

  constexpr void reset() {
    for (size_t i = 0; i < NUMBER_OF_UNITS; ++i)
      Data[i] = 0;
  }

  constexpr bool test(size_t Index) const {
    return Data[Index / BITS_PER_UNIT] & mask(Index);
  }

  constexpr void flip() {
    for (size_t i = 0; i < NUMBER_OF_UNITS; ++i)
      Data[i] = ~Data[i];
  }

  // This function sets all bits in the range from Start to End (inclusive) to
  // true. It assumes that Start <= End.
  constexpr void set_range(size_t Start, size_t End) {
    size_t start_index = Start / BITS_PER_UNIT;
    size_t end_index = End / BITS_PER_UNIT;

    if (start_index == end_index) {
      // The reason the left shift is split into two parts (instead of just left
      // shifting by End - Start + 1) is because when a number is shifted left
      // by 64 then it wraps around to doing nothing, but shifting by 63 and the
      // shifting by 1 correctly shifts away all of the bits.
      size_t bit_mask = (((size_t(1) << (End - Start)) << 1) - 1)
                        << (Start - (start_index * BITS_PER_UNIT));
      Data[start_index] |= bit_mask;
    } else {
      size_t low_bit_mask =
          ~((size_t(1) << (Start - (start_index * BITS_PER_UNIT))) - 1);
      Data[start_index] |= low_bit_mask;

      for (size_t i = start_index + 1; i < end_index; ++i)
        Data[i] = ~size_t(0);

      // Same as above, by splitting the shift the behavior is more consistent.
      size_t high_bit_mask =
          ((size_t(1) << (End - (end_index * BITS_PER_UNIT))) << 1) - 1;
      Data[end_index] |= high_bit_mask;
    }
  }

  constexpr bool operator==(const bitset<NumberOfBits> &other) {
    for (size_t i = 0; i < NUMBER_OF_UNITS; ++i) {
      if (Data[i] != other.Data[i])
        return false;
    }
    return true;
  }

private:
  static constexpr size_t BITS_PER_BYTE = 8;
  static constexpr size_t BITS_PER_UNIT = BITS_PER_BYTE * sizeof(size_t);
  static constexpr size_t NUMBER_OF_UNITS =
      (NumberOfBits + BITS_PER_UNIT - 1) / BITS_PER_UNIT;

  static inline size_t mask(size_t Index) {
    return size_t{1} << (Index % BITS_PER_UNIT);
  }
  size_t Data[NUMBER_OF_UNITS] = {0};
};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_BITSET_H
