//===-- Utilities to convert integral values to string ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
#define LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H

#include "src/__support/CPP/StringView.h"
#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {

template <typename T> class IntegerToString {
  static_assert(cpp::is_integral_v<T>,
                "IntegerToString can only be used with integral types.");

  using UnsignedType = cpp::make_unsigned_t<T>;

  // We size the string buffer using an approximation algorithm:
  //
  //   size = ceil(sizeof(T) * 5 / 2)
  //
  // If sizeof(T) is 1, then size is 3 (actually 3)
  // If sizeof(T) is 2, then size is 5 (actually need 5)
  // If sizeof(T) is 4, then size is 10 (actually need 10)
  // If sizeof(T) is 8, then size is 20 (actually need 20)
  // If sizeof(T) is 16, then size is 40 (actually need 39)
  //
  // NOTE: The ceil operation is actually implemented as
  //     floor(((sizeof(T) * 5) + 1)/2)
  // where floor operation is just integer division.
  //
  // This estimation grows slightly faster than the actual value, but the
  // overhead is small enough to tolerate. In the actual formula below, we
  // add an additional byte to accommodate the '-' sign in case of signed
  // integers.
  static constexpr size_t BUFSIZE =
      (sizeof(T) * 5 + 1) / 2 + (cpp::is_signed<T>() ? 1 : 0);
  char strbuf[BUFSIZE] = {'\0'};
  size_t len = 0;

  constexpr void convert(UnsignedType val) {
    size_t buffptr = BUFSIZE;
    if (val == 0) {
      strbuf[buffptr - 1] = '0';
      --buffptr;
    } else {
      for (; val > 0; --buffptr, val /= 10)
        strbuf[buffptr - 1] = (val % 10) + '0';
    }
    len = BUFSIZE - buffptr;
  }

public:
  constexpr explicit IntegerToString(T val) {
    convert(val < 0 ? UnsignedType(-val) : UnsignedType(val));
    if (val < 0) {
      // This branch will be taken only for negative signed values.
      ++len;
      strbuf[BUFSIZE - len] = '-';
    }
  }

  cpp::StringView str() const {
    return cpp::StringView(strbuf + BUFSIZE - len, len);
  }

  operator cpp::StringView() const { return str(); }
};

template <typename T> IntegerToString<T> integer_to_string(T val) {
  return IntegerToString<T>(val);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
