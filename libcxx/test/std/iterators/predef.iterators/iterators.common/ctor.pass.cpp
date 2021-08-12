//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr common_iterator() requires default_initializable<I> = default;
// constexpr common_iterator(I i);
// constexpr common_iterator(S s);
// template<class I2, class S2>
//   requires convertible_to<const I2&, I> && convertible_to<const S2&, S>
//     constexpr common_iterator(const common_iterator<I2, S2>& x);

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

template<class I, class S>
concept ValidCommonIterator = requires {
  typename std::common_iterator<I, S>;
};

template<class I, class I2>
concept ConvCtorEnabled = requires(std::common_iterator<I2, sentinel_type<int*>> ci) {
  std::common_iterator<I, sentinel_type<int*>>(ci);
};

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  static_assert( std::is_default_constructible_v<std::common_iterator<int*, sentinel_type<int*>>>);
  static_assert(!std::is_default_constructible_v<std::common_iterator<non_default_constructible_iterator<int*>, sentinel_type<int*>>>);

  // Not copyable:
  static_assert(!ValidCommonIterator<cpp20_input_iterator<int*>, sentinel_type<int*>>);
  // Same iter and sent:
  static_assert(!ValidCommonIterator<cpp17_input_iterator<int*>, cpp17_input_iterator<int*>>);

  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);
    assert(commonIter1 != commonSent1);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);
    assert(commonIter1 != commonSent1);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);
    assert(commonIter1 != commonSent1);
  }

  // Conversion constructor:
  {
    convertible_iterator<int*> conv{buffer};
    auto commonIter1 = std::common_iterator<convertible_iterator<int*>, sentinel_type<int*>>(conv);
    auto commonIter2 = std::common_iterator<forward_iterator<int*>, sentinel_type<int*>>(commonIter1);
    assert(*commonIter2 == 1);

    static_assert( ConvCtorEnabled<forward_iterator<int*>, convertible_iterator<int*>>);
    static_assert(!ConvCtorEnabled<forward_iterator<int*>, random_access_iterator<int*>>);
  }
}

int main(int, char**) {
  test();

  return 0;
}
