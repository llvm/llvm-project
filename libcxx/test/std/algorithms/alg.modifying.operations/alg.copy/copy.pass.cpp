//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy(InIter first, InIter last, OutIter result);

#include <algorithm>
#include <cassert>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

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

template <class InIter>
struct Test {
  template <class OutIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    const unsigned N = 1000;
    int ia[N]        = {};
    for (unsigned i = 0; i < N; ++i)
      ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy(InIter(ia), InIter(ia + N), OutIter(ib));
    assert(base(r) == ib + N);
    for (unsigned i = 0; i < N; ++i)
      assert(ia[i] == ib[i]);
  }
};

struct TestInIters {
  template <class InIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(
        types::concatenate_t<types::cpp17_input_iterator_list<int*>, types::type_list<cpp17_output_iterator<int*> > >(),
        Test<InIter>());
  }
};

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy with aligned bytes
    std::vector<bool> out(N);
    std::copy(in.begin(), in.end(), out.begin());
    assert(in == out);
  }
  { // Test copy with unaligned bytes
    std::vector<bool> out(N + 8);
    std::copy(in.begin(), in.end(), out.begin() + 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<const int*>(), TestInIters());

  { // Make sure that padding bits aren't copied
    Derived src(1, 2, 3);
    Derived dst(4, 5, 6);
    std::copy(static_cast<PaddedBase*>(&src), static_cast<PaddedBase*>(&src) + 1, static_cast<PaddedBase*>(&dst));
    assert(dst.a_ == 1);
    assert(dst.b_ == 2);
    assert(dst.c_ == 6);
  }
  { // Make sure that overlapping ranges can be copied
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::copy(a + 3, a + 10, a);
    int expected[] = {4, 5, 6, 7, 8, 9, 10, 8, 9, 10};
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

  // Validate std::copy with std::vector<bool> iterators and custom storage types.
  // Ensure that assigned bits hold the intended values, while unassigned bits stay unchanged.
  // Related issue: https://llvm.org/PR131692.
  {
    //// Tests for std::copy with aligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy(in.begin() + 1, in.begin() + 2, out.begin() + 1); // out[1] = false
      assert(out[1] == false);
      for (std::size_t i = 0; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        if (i != 1)
          assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy(in.begin(), in.begin() + 1, out.begin()); // out[0] = false
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test middle (whole) words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(32, true, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = false;
      std::vector<bool, Alloc> out(32, false, Alloc(1));
      std::copy(in.begin() + 4, in.end() - 4, out.begin() + 4);
      for (std::size_t i = 4; i < static_cast<std::size_t>(in.size() - 4); ++i)
        assert(in[i] == out[i]);
      for (std::size_t i = 0; i < 4; ++i)
        assert(out[i] == false);
      for (std::size_t i = 28; i < out.size(); ++i)
        assert(out[i] == false);
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy(in.begin() + 1, in.begin() + 3, out.begin() + 1); // out[1..2] = false
      assert(out[1] == false);
      assert(out[2] == false);
      for (std::size_t i = 0; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        if (i != 1 && i != 2)
          assert(out[i] == true);
    }
    { // Test the last (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy(in.begin(), in.begin() + 2, out.begin()); // out[0..1] = false
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test middle (whole) words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(64, true, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = false;
      std::vector<bool, Alloc> out(64, false, Alloc(1));
      std::copy(in.begin() + 8, in.end() - 8, out.begin() + 8);
      for (std::size_t i = 8; i < static_cast<std::size_t>(in.size() - 8); ++i)
        assert(in[i] == out[i]);
      for (std::size_t i = 0; i < 8; ++i)
        assert(out[i] == false);
      for (std::size_t i = static_cast<std::size_t>(out.size() - 8); i < out.size(); ++i)
        assert(out[i] == false);
    }

    //// Tests for std::copy with unaligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy(in.begin() + 7, in.end(), out.begin()); // out[0] = false
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::copy(in.begin(), in.begin() + 1, out.begin() + 2); // out[2] = false
      assert(out[2] == false);
      for (std::size_t i = 1; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        if (i != 2)
          assert(out[i] == true);
    }
    { // Test middle (whole) words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(36, true, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = false;
      std::vector<bool, Alloc> out(40, false, Alloc(1));
      std::copy(in.begin(), in.end(), out.begin() + 4);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 4]);
      for (std::size_t i = 0; i < 4; ++i)
        assert(out[i] == false);
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy(in.begin() + 14, in.end(), out.begin()); // out[0..1] = false
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::copy(in.begin(), in.begin() + 2, out.begin() + 1); // out[1..2] = false
      assert(out[1] == false);
      assert(out[2] == false);
      for (std::size_t i = 0; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        if (i != 1 && i != 2)
          assert(out[i] == true);
    }
    { // Test middle (whole) words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(72, true, Alloc(1));
      for (std::size_t i = 0; i < in.size(); i += 2)
        in[i] = false;
      std::vector<bool, Alloc> out(80, false, Alloc(1));
      std::copy(in.begin(), in.end(), out.begin() + 4);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 4]);
      for (std::size_t i = 0; i < 4; ++i)
        assert(out[i] == false);
      for (std::size_t i = in.size() + 4; i < out.size(); ++i)
        assert(out[i] == false);
    }
  }

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
