//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void shrink_to_fit();

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <climits>
#include <vector>

#include "increasing_allocator.h"
#include "min_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    using C = std::vector<bool>;
    C v(100);
    v.push_back(1);
    C::size_type before_cap = v.capacity();
    v.clear();
    v.shrink_to_fit();
    assert(v.capacity() <= before_cap);
    LIBCPP_ASSERT(v.capacity() == 0); // libc++ honors the shrink_to_fit request as a QOI matter
    assert(v.size() == 0);
  }
  {
    using C = std::vector<bool, min_allocator<bool> >;
    C v(100);
    v.push_back(1);
    C::size_type before_cap = v.capacity();
    v.shrink_to_fit();
    assert(v.capacity() >= 101);
    assert(v.capacity() <= before_cap);
    assert(v.size() == 101);
    v.erase(v.begin() + 1, v.end());
    v.shrink_to_fit();
    assert(v.capacity() <= before_cap);
    LIBCPP_ASSERT(v.capacity() == C(1).capacity()); // libc++ honors the shrink_to_fit request as a QOI matter.
    assert(v.size() == 1);
  }

#if defined(_LIBCPP_VERSION)
  {
    using C                = std::vector<bool>;
    unsigned bits_per_word = static_cast<unsigned>(sizeof(C::__storage_type) * CHAR_BIT);
    C v(bits_per_word);
    v.push_back(1);
    assert(v.capacity() == bits_per_word * 2);
    assert(v.size() == bits_per_word + 1);
    v.pop_back();
    v.shrink_to_fit();
    assert(v.capacity() == bits_per_word);
    assert(v.size() == bits_per_word);
  }
  {
    using C                = std::vector<bool>;
    unsigned bits_per_word = static_cast<unsigned>(sizeof(C::__storage_type) * CHAR_BIT);
    C v;
    v.reserve(bits_per_word * 2);
    v.push_back(1);
    assert(v.capacity() == bits_per_word * 2);
    assert(v.size() == 1);
    v.shrink_to_fit();
    assert(v.capacity() == bits_per_word);
    assert(v.size() == 1);
  }
#endif

  return true;
}

#if TEST_STD_VER >= 23
// https://llvm.org/PR95161
constexpr bool test_increasing_allocator() {
  std::vector<bool, increasing_allocator<bool>> v;
  v.push_back(1);
  std::size_t capacity = v.capacity();
  v.shrink_to_fit();
  assert(v.capacity() <= capacity);
  assert(v.size() == 1);

  return true;
}
#endif // TEST_STD_VER >= 23

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
#if TEST_STD_VER >= 23
  test_increasing_allocator();
  static_assert(test_increasing_allocator());
#endif // TEST_STD_VER >= 23

  return 0;
}
