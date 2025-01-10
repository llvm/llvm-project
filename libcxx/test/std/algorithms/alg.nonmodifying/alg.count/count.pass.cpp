//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, class T>
//   requires HasEqualTo<Iter::value_type, T>
//   constexpr Iter::difference_type   // constexpr after C++17
//   count(Iter first, Iter last, const T& value);

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=20000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=80000000
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(std::count(Iter(ia), Iter(ia + sa), 2) == 3);
    assert(std::count(Iter(ia), Iter(ia + sa), 7) == 0);
    assert(std::count(Iter(ia), Iter(ia), 2) == 0);
  }
};

TEST_CONSTEXPR_CXX20 void test_bit_iterator_with_custom_sized_types() {
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> in(100, true, Alloc(1));
    assert(std::count(in.begin(), in.end(), true) == 100);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> in(199, true, Alloc(1));
    assert(std::count(in.begin(), in.end(), true) == 199);
  }
  {
    using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
    std::vector<bool, Alloc> in(200, true, Alloc(1));
    assert(std::count(in.begin(), in.end(), true) == 200);
  }
  {
    using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
    std::vector<bool, Alloc> in(257, true, Alloc(1));
    assert(std::count(in.begin(), in.end(), true) == 257);
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<const int*>(), Test());

  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED) {
    std::vector<bool> vec(256 + 64);
    for (ptrdiff_t i = 0; i != 256; ++i) {
      for (size_t offset = 0; offset != 64; ++offset) {
        std::fill(vec.begin(), vec.end(), false);
        std::fill(vec.begin() + offset, vec.begin() + i + offset, true);
        assert(std::count(vec.begin() + offset, vec.begin() + offset + 256, true) == i);
        assert(std::count(vec.begin() + offset, vec.begin() + offset + 256, false) == 256 - i);
      }
    }
  }

  test_bit_iterator_with_custom_sized_types();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
