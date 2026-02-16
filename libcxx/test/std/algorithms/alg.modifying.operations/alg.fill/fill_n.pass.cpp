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
#include <deque>
#include <ranges>
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

struct source {
  TEST_CONSTEXPR source() = default;
  TEST_CONSTEXPR_CXX20 operator int() const { return 1; }
};

class CharWrapper {
  char a_;

public:
  TEST_CONSTEXPR CharWrapper() : a_('a') {};
  TEST_CONSTEXPR explicit CharWrapper(char a) : a_(a) {}
  TEST_CONSTEXPR operator unsigned char() const { return 'b'; }

  TEST_CONSTEXPR friend bool operator==(const CharWrapper& x, const CharWrapper& y) { return x.a_ == y.a_; }
};

struct CharTransformer {
  TEST_CONSTEXPR CharTransformer() : c(0) {}
  TEST_CONSTEXPR CharTransformer(char xc) : c(xc + 1) {}
  char c;
};

struct CharUnionStorage {
  union {
    unsigned char a;
    unsigned char b;
  };
};

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  {   // Test cases validating leading/trailing bits unfilled remain unchanged
    { // Leading bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = false;
      std::fill_n(in.begin() + 2, N - 2, true);
      assert(in == expected);
    }
    { // Trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[N - 1] = expected[N - 2] = false;
      std::fill_n(in.begin(), N - 2, true);
      assert(in == expected);
    }
    { // Leading and trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = expected[N - 1] = expected[N - 2] = false;
      std::fill_n(in.begin() + 2, N - 4, true);
      assert(in == expected);
    }
  }

  {   // Test cases with full or partial bytes filled
    { // Full bytes filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::fill_n(in.begin(), N, true);
      assert(in == expected);
    }
    { // Partial bytes with offset filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::fill_n(in.begin() + 4, N - 8, true);
      std::fill_n(expected.begin(), 4, false);
      std::fill_n(expected.end() - 4, 4, false);
      assert(in == expected);
    }
  }

  return true;
}

/*TEST_CONSTEXPR_CXX26*/ void test_deque() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  std::deque<int> in(20);
  std::deque<int> expected(in.size(), 42);
  std::fill_n(in.begin(), in.size(), 42);
  assert(in == expected);
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<char*>(), Test<char>());
  types::for_each(types::forward_iterator_list<int*>(), Test<int>());

  { // Test with int arrays
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

  { // Test with struct arrays
    {
      CharWrapper a[3];
      assert(std::fill_n(&a[0], UDI(3), CharWrapper('a')) == a + 3);
      assert(a[0] == CharWrapper('a'));
      assert(a[1] == CharWrapper('a'));
      assert(a[2] == CharWrapper('a'));
    }
    {
      CharTransformer b[4] = {};
      assert(std::fill_n(b, UDI(4), static_cast<char>(10)) == b + 4);
      assert(b[0].c == 11);
      assert(b[1].c == 11);
      assert(b[2].c == 11);
      assert(b[3].c == 11);
    }
    {
      CharUnionStorage foo[5];
      std::fill_n(&foo[0], UDI(5), CharUnionStorage());
    }
  }

  { // Test with an int array and struct source
    int a[4] = {};
    assert(std::fill_n(a, UDI(4), source()) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
  }

  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));

    // Make sure std::fill_n behaves properly with std::vector<bool> iterators with custom size types.
    // See https://github.com/llvm/llvm-project/pull/122410.
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

  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_deque();

#if TEST_STD_VER >= 20
  {
    std::vector<std::vector<int>> v{{1, 2}, {1, 2, 3}, {}, {3, 4, 5}, {6}, {7, 8, 9, 6}, {0, 1, 2, 3, 0, 1, 2}};
    auto jv = std::ranges::join_view(v);
    std::fill_n(jv.begin(), std::distance(jv.begin(), jv.end()), 42);
    for (const auto& vec : v)
      for (auto n : vec)
        assert(n == 42);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
