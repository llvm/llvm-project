//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr OutputIterator      // constexpr after C++17
//   fill_n(Iter first, Size n, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"
#include "user_defined_integral.h"

typedef UserDefinedIntegral<unsigned> UDI;

template <class Iter, class Container>
TEST_CONSTEXPR_CXX20 void
test(Container in, size_t from, size_t n, typename Container::value_type value, Container expected) {
  Iter it = std::fill_n(Iter(in.data() + from), UDI(n), value);
  assert(base(it) == in.data() + from + n);
  assert(in == expected);
}

template <class T>
struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    {
      std::array<T, 4> in       = {1, 2, 3, 4};
      std::array<T, 4> expected = {5, 5, 5, 5};
      test<Iter>(in, 0, 4, 5, expected);
    }
    {
      std::array<T, 4> in       = {1, 2, 3, 4};
      std::array<T, 4> expected = {1, 5, 5, 4};
      test<Iter>(in, 1, 2, 5, expected);
    }
  }
};

TEST_CONSTEXPR_CXX20 void test_int_array() {
  {
    int a[4] = {};
    assert(std::fill_n(a, UDI(4), static_cast<char>(1)) == a + 4);
    assert(a[0] == 1 && a[1] == 1 && a[2] == 1 && a[3] == 1);
  }
#if TEST_STD_VER >= 11
  {
    const std::size_t N = 5;
    int ib[]            = {0, 0, 0, 0, 0, 0}; // one bigger than N

    auto it = std::fill_n(std::begin(ib), N, 5);
    assert(it == (std::begin(ib) + N) && std::all_of(std::begin(ib), it, [](int a) { return a == 5; }) &&
           *it == 0 // don't overwrite the last value in the output array
    );
  }
#endif
}

struct source {
  TEST_CONSTEXPR source() = default;
  TEST_CONSTEXPR_CXX20 operator int() const { return 1; }
};

TEST_CONSTEXPR_CXX20 void test_int_array_struct_source() {
  int a[4] = {};
  assert(std::fill_n(a, UDI(4), source()) == a + 4);
  assert(a[0] == 1);
  assert(a[1] == 1);
  assert(a[2] == 1);
  assert(a[3] == 1);
}

class A {
  char a_;

public:
  TEST_CONSTEXPR A() : a_('a') {};
  TEST_CONSTEXPR explicit A(char a) : a_(a) {}
  TEST_CONSTEXPR operator unsigned char() const { return 'b'; }

  TEST_CONSTEXPR friend bool operator==(const A& x, const A& y) { return x.a_ == y.a_; }
};

struct B {
  TEST_CONSTEXPR B() : c(0) {}
  TEST_CONSTEXPR B(char xc) : c(xc + 1) {}
  char c;
};

struct Storage {
  union {
    unsigned char a;
    unsigned char b;
  };
};

// Make sure std::fill_n behaves properly with std::vector<bool> iterators with custom size types.
// See https://github.com/llvm/llvm-project/pull/122410.
TEST_CONSTEXPR_CXX20 void test_bititer_with_custom_sized_types() {
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> in(100, false, Alloc(1));
    std::vector<bool, Alloc> expected(100, true, Alloc(1));
    std::fill_n(in.begin(), in.size(), true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::fill_n(in.begin(), in.size(), true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::fill_n(in.begin(), in.size(), true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::fill_n(in.begin(), in.size(), true);
    assert(in == expected);
  }
}

TEST_CONSTEXPR_CXX20 void test_struct_array() {
  {
    A a[3];
    assert(std::fill_n(&a[0], UDI(3), A('a')) == a + 3);
    assert(a[0] == A('a'));
    assert(a[1] == A('a'));
    assert(a[2] == A('a'));
  }
  {
    B b[4] = {};
    assert(std::fill_n(b, UDI(4), static_cast<char>(10)) == b + 4);
    assert(b[0].c == 11);
    assert(b[1].c == 11);
    assert(b[2].c == 11);
    assert(b[3].c == 11);
  }
  {
    Storage foo[5];
    std::fill_n(&foo[0], UDI(5), Storage());
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<char*>(), Test<char>());
  types::for_each(types::forward_iterator_list<int*>(), Test<int>());

  test_int_array();
  test_struct_array();
  test_int_array_struct_source();
  test_bititer_with_custom_sized_types();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
