//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator InIter, BidirectionalIterator OutIter>
//   requires OutputIterator<OutIter, InIter::reference>
//   constexpr OutIter   // constexpr after C++17
//   copy_backward(InIter first, InIter last, OutIter result);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"
#include "user_defined_integral.h"

class PaddedBase {
public:
  TEST_CONSTEXPR PaddedBase(std::int16_t a, std::int8_t b) : a_(a), b_(b) {}

  std::int16_t a_;
  std::int8_t b_;
};

class Derived : public PaddedBase {
public:
  TEST_CONSTEXPR Derived(std::int16_t a, std::int8_t b, std::int8_t c) : PaddedBase(a, b), c_(c) {}

  std::int8_t c_;
};

struct TestIterators {
  template <class InIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::bidirectional_iterator_list<int*>(), TestImpl<InIter>());
  }

  template <class InIter>
  struct TestImpl {
    template <class OutIter>
    TEST_CONSTEXPR_CXX20 void operator()() {
      const unsigned N = 1000;
      int ia[N]        = {};
      for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
      int ib[N] = {0};

      OutIter r = std::copy_backward(InIter(ia), InIter(ia + N), OutIter(ib + N));
      assert(base(r) == ib);
      for (unsigned i = 0; i < N; ++i)
        assert(ia[i] == ib[i]);
    }
  };
};

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy_backward with aligned bytes
    std::vector<bool> out(N);
    std::copy_backward(in.begin(), in.end(), out.end());
    assert(in == out);
  }
  { // Test copy_backward with unaligned bytes
    std::vector<bool> out(N + 8);
    std::copy_backward(in.begin(), in.end(), out.end() - 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::bidirectional_iterator_list<const int*>(), TestIterators());

  { // Make sure that padding bits aren't copied
    Derived src(1, 2, 3);
    Derived dst(4, 5, 6);
    std::copy_backward(
        static_cast<PaddedBase*>(&src), static_cast<PaddedBase*>(&src) + 1, static_cast<PaddedBase*>(&dst) + 1);
    assert(dst.a_ == 1);
    assert(dst.b_ == 2);
    assert(dst.c_ == 6);
  }

  { // Make sure that overlapping ranges can be copied
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::copy_backward(a, a + 7, a + 10);
    int expected[] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 7};
    assert(std::equal(a, a + 10, expected));
  }

  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));
  }

  // Validate std::copy_backward with std::vector<bool> iterators and custom storage types.
  // Ensure that assigned bits hold the intended values, while unassigned bits stay unchanged.
  // Related issue: https://llvm.org/PR131718.
  {
    //// Tests for std::copy_backward with aligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(7, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy_backward(in.begin(), in.begin() + 1, out.begin() + 1);
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy_backward(in.end() - 4, in.end(), out.end());
      for (std::size_t i = 0; i < static_cast<std::size_t>(in.size() - 4); ++i)
        assert(out[i] == true);
      for (std::size_t i = in.size() + 4; i < out.size(); ++i)
        assert(in[i] == out[i]);
    }
    { // Test the middle (whole) words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(17, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(24, true, Alloc(1));
      std::copy_backward(in.begin(), in.end(), out.begin() + in.size());
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i]);
      for (std::size_t i = in.size(); i < out.size(); ++i)
        assert(out[i] == true);
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(14, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy_backward(in.begin(), in.begin() + 2, out.begin() + 2);
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy_backward(in.end() - 8, in.end(), out.end());
      for (std::size_t i = 0; i < static_cast<std::size_t>(in.size() - 8); ++i)
        assert(out[i] == true);
      for (std::size_t i = in.size() + 8; i < out.size(); ++i)
        assert(in[i] == out[i]);
    }
    { // Test the middle (whole) words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(34, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(48, true, Alloc(1));
      std::copy_backward(in.begin(), in.end(), out.begin() + in.size());
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i]);
      for (std::size_t i = in.size(); i < out.size(); ++i)
        assert(out[i] == true);
    }

    //// Tests for std::copy_backward with unaligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy_backward(in.begin(), in.begin() + 1, out.begin() + 1);
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy_backward(in.end() - 1, in.end(), out.begin() + 1);
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the middle (whole) words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(17, true, Alloc(1));
      std::copy_backward(in.begin(), in.end(), out.end());
      assert(out[0] == true);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 1]);
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy_backward(in.begin(), in.begin() + 2, out.begin() + 2);
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy_backward(in.end() - 2, in.end(), out.begin() + 2);
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i)
        assert(out[i] == true);
    }
    { // Test the middle (whole) words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(32, false, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = true;
      std::vector<bool, Alloc> out(33, true, Alloc(1));
      std::copy_backward(in.begin(), in.end(), out.end());
      assert(out[0] == true);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 1]);
    }
  }

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
