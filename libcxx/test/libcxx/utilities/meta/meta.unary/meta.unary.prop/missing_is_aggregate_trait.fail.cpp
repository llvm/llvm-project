//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <type_traits>

// template <class T> struct is_aggregate;
// template <class T> constexpr bool is_aggregate_v = is_aggregate<T>::value;

#include <type_traits>

int main ()
{
#ifdef _LIBCPP_HAS_NO_IS_AGGREGATE
  // This should not compile when _LIBCPP_HAS_NO_IS_AGGREGATE is defined.
  bool b = __is_aggregate(void);
  ((void)b);
#else
#error Forcing failure...
#endif
}
