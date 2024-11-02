//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(); // Deprecated in C++20.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(typename S::size_type min_cap, typename S::size_type erased_index)
{
    S s(min_cap, 'a');
    s.erase(erased_index);
    assert(s.size() == erased_index);
    assert(s.capacity() >= min_cap); // Check that we really have at least this capacity.

    typename S::size_type old_cap = s.capacity();
    S s0 = s;
    s.reserve();
    LIBCPP_ASSERT(s.__invariants());
    assert(s == s0);
    assert(s.capacity() <= old_cap);
    assert(s.capacity() >= s.size());
}

template <class S>
void test_string() {
  test<S>(0, 0);
  test<S>(10, 5);
  test<S>(100, 50);
}

bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
