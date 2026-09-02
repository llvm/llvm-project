//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <__utility/pointer_int_pair.h>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

template <class Ptr, class UnderlyingType>
using single_bit_pair = std::__pointer_int_pair<Ptr, UnderlyingType, std::__integer_width(1)>;

template <class Ptr, class UnderlyingType>
using two_bit_pair = std::__pointer_int_pair<Ptr, UnderlyingType, std::__integer_width(2)>;

#if _LIBCPP_STD_VER >= 20
constinit single_bit_pair<int*, size_t> continitiable_pointer_int_pair;
#endif

int main(int, char**) {
#if TEST_STD_VER >= 11
  { // __pointer_int_pair() constructor
    single_bit_pair<int*, size_t> pair = {};
    assert(pair.__get_value() == 0);
    assert(pair.__get_ptr() == nullptr);
  }
#endif

  { // __pointer_int_pair(pointer, int) constructor
    single_bit_pair<int*, size_t> pair(nullptr, 1);
    assert(pair.__get_value() == 1);
    assert(pair.__get_ptr() == nullptr);
  }

  { // pointer is correctly packed/unpacked (with different types and values)
    int i;
    single_bit_pair<int*, size_t> pair(&i, 0);
    assert(pair.__get_value() == 0);
    assert(pair.__get_ptr() == &i);
  }
  {
    int i;
    two_bit_pair<int*, size_t> pair(&i, 2);
    assert(pair.__get_value() == 2);
    assert(pair.__get_ptr() == &i);
  }
  {
    short i;
    single_bit_pair<short*, size_t> pair(&i, 1);
    assert(pair.__get_value() == 1);
    assert(pair.__get_ptr() == &i);
  }

  { // check that a __pointer_int_pair<__pointer_int_pair> works
    int i;
    single_bit_pair<single_bit_pair<int*, size_t>, size_t> pair(single_bit_pair<int*, size_t>(&i, 1), 0);
    assert(pair.__get_value() == 0);
    assert(pair.__get_ptr().__get_ptr() == &i);
    assert(pair.__get_ptr().__get_value() == 1);
  }

#if _LIBCPP_STD_VER >= 17
  { // check that the tuple protocol is correctly implemented
    int i;
    two_bit_pair<int*, size_t> pair{&i, 3};
    auto [ptr, value] = pair;
    assert(ptr == &i);
    assert(value == 3);
  }
#endif

  { // check that the (pointer, int) constructor is implicit
    int i;
    two_bit_pair<int*, size_t> pair(&i, 3);
    assert(pair.__get_ptr() == &i);
    assert(pair.__get_value() == 3);
  }

  { // check that overaligned types work as expected
    struct TEST_ALIGNAS(32) Overaligned {
      int i;
    };

    Overaligned i;
    std::__pointer_int_pair<Overaligned*, size_t, std::__integer_width(4)> pair(&i, 13);
    assert(pair.__get_ptr() == &i);
    assert(pair.__get_value() == 13);
  }

  { // check that types other than size_t work as well
    int i;
    single_bit_pair<int*, bool> pair(&i, true);
    assert(pair.__get_ptr() == &i);
    assert(pair.__get_value());
    static_assert(std::is_same<decltype(pair.__get_value()), bool>::value, "");
  }
  {
    int i;
    single_bit_pair<int*, unsigned char> pair(&i, 1);
    assert(pair.__get_ptr() == &i);
    assert(pair.__get_value() == 1);
    static_assert(std::is_same<decltype(pair.__get_value()), unsigned char>::value, "");
  }

  return 0;
}
