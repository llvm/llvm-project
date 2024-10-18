//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Make sure the size we allocate and deallocate match. See https://github.com/llvm/llvm-project/pull/90292.

#include <string>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

static int allocated_;

template <class T, class Sz>
struct test_alloc {
  typedef Sz size_type;
  typedef typename std::make_signed<Sz>::type difference_type;
  typedef T value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef typename std::add_lvalue_reference<value_type>::type reference;
  typedef typename std::add_lvalue_reference<const value_type>::type const_reference;

  template <class U>
  struct rebind {
    typedef test_alloc<U, Sz> other;
  };

  TEST_CONSTEXPR_CXX14 pointer allocate(size_type n, const void* = nullptr) {
    allocated_ += n;
    return std::allocator<value_type>().allocate(n);
  }

  TEST_CONSTEXPR_CXX14 void deallocate(pointer p, size_type s) {
    allocated_ -= s;
    std::allocator<value_type>().deallocate(p, s);
  }
};

template <class Sz>
void test() {
  for (int i = 1; i < 1000; ++i) {
    using Str = std::basic_string<char, std::char_traits<char>, test_alloc<char, Sz> >;
    {
      Str s(i, 't');
      assert(allocated_ == 0 || allocated_ >= i);
    }
  }
  assert(allocated_ == 0);
}

int main(int, char**) {
  test<uint32_t>();
  test<uint64_t>();
  test<size_t>();

  return 0;
}
