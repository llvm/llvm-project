//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::copy_result<I, O> ranges::copy(I first, S last, O result);
// template<input_range R, weakly_incrementable O>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr ranges::copy_result<borrowed_iterator_t<R>, O> ranges::copy(R&& r, O result);

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "sized_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class In, class Out = In, class Sent = sentinel_wrapper<In>>
concept HasCopyIt = requires(In in, Sent sent, Out out) { std::ranges::copy(in, sent, out); };

static_assert(HasCopyIt<int*>);
static_assert(!HasCopyIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCopyIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCopyIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCopyIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyCopyable {};
static_assert(!HasCopyIt<int*, NotIndirectlyCopyable*>);
static_assert(!HasCopyIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasCopyIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range, class Out>
concept HasCopyR = requires(Range range, Out out) { std::ranges::copy(range, out); };

static_assert(HasCopyR<std::array<int, 10>, int*>);
static_assert(!HasCopyR<InputRangeNotDerivedFrom, int*>);
static_assert(!HasCopyR<InputRangeNotIndirectlyReadable, int*>);
static_assert(!HasCopyR<InputRangeNotInputOrOutputIterator, int*>);
static_assert(!HasCopyR<WeaklyIncrementableNotMovable, int*>);
static_assert(!HasCopyR<UncheckedRange<NotIndirectlyCopyable*>, int*>);
static_assert(!HasCopyR<InputRangeNotSentinelSemiregular, int*>);
static_assert(!HasCopyR<InputRangeNotSentinelEqualityComparableWith, int*>);

static_assert(std::is_same_v<std::ranges::copy_result<int, long>, std::ranges::in_out_result<int, long>>);

// clang-format off
template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  { // simple test
    {
      std::array in{1, 2, 3, 4};
      std::array<int, 4> out;
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
          std::ranges::copy(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data()));
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
    {
      std::array in{1, 2, 3, 4};
      std::array<int, 4> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret = std::ranges::copy(range, Out(out.data()));
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto ret = std::ranges::copy(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data()));
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      auto ret   = std::ranges::copy(range, Out(out.data()));
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
  }
}
// clang-format on

#if TEST_STD_VER >= 23
constexpr bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy with aligned bytes
    std::vector<bool> out(N);
    std::ranges::copy(in, out.begin());
    assert(in == out);
  }
  { // Test copy with unaligned bytes
    std::vector<bool> out(N + 8);
    std::ranges::copy(in, out.begin() + 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }

  return true;
}
#endif

constexpr bool test() {
  types::for_each(types::forward_iterator_list<int*>{}, []<class Out>() {
    test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
    test_iterators<ProxyIterator<cpp20_input_iterator<int*>>,
                   ProxyIterator<Out>,
                   sentinel_wrapper<ProxyIterator<cpp20_input_iterator<int*>>>>();

    types::for_each(types::forward_iterator_list<int*>{}, []<class In>() {
      test_iterators<In, Out>();
      test_iterators<In, Out, sized_sentinel<In>>();
      test_iterators<In, Out, sentinel_wrapper<In>>();

      test_iterators<ProxyIterator<In>, ProxyIterator<Out>>();
      test_iterators<ProxyIterator<In>, ProxyIterator<Out>, sized_sentinel<ProxyIterator<In>>>();
      test_iterators<ProxyIterator<In>, ProxyIterator<Out>, sentinel_wrapper<ProxyIterator<In>>>();
    });
  });

  { // check that ranges::dangling is returned
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
        std::ranges::copy(std::array{1, 2, 3, 4}, out.data());
    assert(ret.out == out.data() + 4);
    assert((out == std::array{1, 2, 3, 4}));
  }

  { // check that an iterator is returned with a borrowing range
    std::array in{1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::array<int, 4>::iterator, int*>> auto ret =
        std::ranges::copy(std::views::all(in), out.data());
    assert(ret.in == in.end());
    assert(ret.out == out.data() + 4);
    assert(in == out);
  }

  { // check that every element is copied exactly once
    struct CopyOnce {
      bool copied                               = false;
      constexpr CopyOnce()                      = default;
      constexpr CopyOnce(const CopyOnce& other) = delete;
      constexpr CopyOnce& operator=(const CopyOnce& other) {
        assert(!other.copied);
        copied = true;
        return *this;
      }
    };
    {
      std::array<CopyOnce, 4> in{};
      std::array<CopyOnce, 4> out{};
      auto ret = std::ranges::copy(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
    {
      std::array<CopyOnce, 4> in{};
      std::array<CopyOnce, 4> out{};
      auto ret = std::ranges::copy(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
  }

  { // check that the range is copied forwards
    struct OnlyForwardsCopyable {
      OnlyForwardsCopyable* next = nullptr;
      bool canCopy               = false;
      OnlyForwardsCopyable()     = default;
      constexpr OnlyForwardsCopyable& operator=(const OnlyForwardsCopyable&) {
        assert(canCopy);
        if (next != nullptr)
          next->canCopy = true;
        return *this;
      }
    };
    {
      std::array<OnlyForwardsCopyable, 3> in{};
      std::array<OnlyForwardsCopyable, 3> out{};
      out[0].next    = &out[1];
      out[1].next    = &out[2];
      out[0].canCopy = true;
      auto ret       = std::ranges::copy(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].canCopy);
      assert(out[1].canCopy);
      assert(out[2].canCopy);
    }
    {
      std::array<OnlyForwardsCopyable, 3> in{};
      std::array<OnlyForwardsCopyable, 3> out{};
      out[0].next    = &out[1];
      out[1].next    = &out[2];
      out[0].canCopy = true;
      auto ret       = std::ranges::copy(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].canCopy);
      assert(out[1].canCopy);
      assert(out[2].canCopy);
    }
  }

#if TEST_STD_VER >= 23
  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));
  }

  // Validate std::ranges::copy with std::vector<bool> iterators and custom storage types.
  // Ensure that assigned bits hold the intended values, while unassigned bits stay unchanged.
  // Related issue: https://llvm.org/PR131692.
  {
    //// Tests for std::ranges::copy with aligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin() + 1, in.begin() + 2), out.begin() + 1); // out[1] = false
      assert(out[1] == false);
      for (std::size_t i = 0; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        if (i != 1)
          assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin(), in.begin() + 1), out.begin()); // out[0] = false
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
      std::ranges::copy(std::ranges::subrange(in.begin() + 4, in.end() - 4), out.begin() + 4);
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
      std::ranges::copy(std::ranges::subrange(in.begin() + 1, in.begin() + 3), out.begin() + 1); // out[1..2] = false
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
      std::ranges::copy(std::ranges::subrange(in.begin(), in.begin() + 2), out.begin()); // out[0..1] = false
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
      std::ranges::copy(std::ranges::subrange(in.begin() + 8, in.end() - 8), out.begin() + 8);
      for (std::size_t i = 8; i < static_cast<std::size_t>(in.size() - 8); ++i)
        assert(in[i] == out[i]);
      for (std::size_t i = 0; i < 8; ++i)
        assert(out[i] == false);
      for (std::size_t i = static_cast<std::size_t>(out.size() - 8); i < out.size(); ++i)
        assert(out[i] == false);
    }

    //// Tests for std::ranges::copy with unaligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin() + 7, in.end()), out.begin()); // out[0] = false
      assert(out[0] == false);
      for (std::size_t i = 1; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(8, false, Alloc(1));
      std::vector<bool, Alloc> out(8, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin(), in.begin() + 1), out.begin() + 2); // out[2] = false
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
      std::ranges::copy(std::ranges::subrange(in.begin(), in.end()), out.begin() + 4);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 4]);
      for (std::size_t i = 0; i < 4; ++i)
        assert(out[i] == false);
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin() + 14, in.end()), out.begin()); // out[0..1] = false
      assert(out[0] == false);
      assert(out[1] == false);
      for (std::size_t i = 2; i < out.size(); ++i) // Ensure that unassigned bits remain unchanged
        assert(out[i] == true);
    }
    { // Test the last (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(16, false, Alloc(1));
      std::vector<bool, Alloc> out(16, true, Alloc(1));
      std::ranges::copy(std::ranges::subrange(in.begin(), in.begin() + 2), out.begin() + 1); // out[1..2] = false
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
      std::ranges::copy(std::ranges::subrange(in.begin(), in.end()), out.begin() + 4);
      for (std::size_t i = 0; i < in.size(); ++i)
        assert(in[i] == out[i + 4]);
      for (std::size_t i = 0; i < 4; ++i)
        assert(out[i] == false);
      for (std::size_t i = in.size() + 4; i < out.size(); ++i)
        assert(out[i] == false);
    }
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
