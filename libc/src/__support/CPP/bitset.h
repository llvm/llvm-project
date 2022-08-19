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

  constexpr bool test(size_t Index) const {
    return Data[Index / BITS_PER_UNIT] & mask(Index);
  }

private:
  static constexpr size_t BITS_PER_BYTE = 8;
  static constexpr size_t BITS_PER_UNIT = BITS_PER_BYTE * sizeof(size_t);

  static inline size_t mask(size_t Index) {
    return size_t{1} << (Index % BITS_PER_UNIT);
  }
  size_t Data[(NumberOfBits + BITS_PER_UNIT - 1) / BITS_PER_UNIT] = {0};
};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_BITSET_H
