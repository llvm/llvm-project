//===-- A self contained equivalent of std::bitset --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_CPP_BITSET_H
#define LLVM_LIBC_UTILS_CPP_BITSET_H

#include <stddef.h> // For size_t.
#include <stdint.h> // For uintptr_t.

namespace __llvm_libc {
namespace cpp {

template <size_t NumberOfBits> struct Bitset {
  static_assert(NumberOfBits != 0,
                "Cannot create a __llvm_libc::cpp::Bitset of size 0.");

  constexpr void set(size_t Index) {
    Data[Index / BitsPerUnit] |= (uintptr_t{1} << (Index % BitsPerUnit));
  }

  constexpr bool test(size_t Index) const {
    return Data[Index / BitsPerUnit] & (uintptr_t{1} << (Index % BitsPerUnit));
  }

private:
  static constexpr size_t BitsPerByte = 8;
  static constexpr size_t BitsPerUnit = BitsPerByte * sizeof(uintptr_t);
  uintptr_t Data[(NumberOfBits + BitsPerUnit - 1) / BitsPerUnit] = {0};
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_BITSET_H
