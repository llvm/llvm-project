//===-- String utils for matchers -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_SIMPLE_STRING_CONV_H
#define LLVM_LIBC_UTILS_UNITTEST_SIMPLE_STRING_CONV_H

#include "src/__support/CPP/TypeTraits.h"

#include <string>

namespace __llvm_libc {

// Return the first N hex digits of an integer as a string in upper case.
template <typename T>
cpp::EnableIfType<cpp::IsIntegral<T>::Value, std::string>
int_to_hex(T X, size_t Length = sizeof(T) * 2) {
  std::string s(Length, '0');

  for (auto it = s.rbegin(), end = s.rend(); it != end; ++it, X >>= 4) {
    unsigned char Mod = static_cast<unsigned char>(X) & 15;
    *it = (Mod < 10 ? '0' + Mod : 'a' + Mod - 10);
  }

  return s;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_UNITTEST_SIMPLE_STRING_CONV_H
