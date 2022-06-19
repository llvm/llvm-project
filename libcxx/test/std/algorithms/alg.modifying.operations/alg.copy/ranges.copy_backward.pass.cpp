//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges


// template<bidirectional_iterator I1, sentinel_for<I1> S1, bidirectional_iterator I2>
//   requires indirectly_copyable<I1, I2>
//   constexpr ranges::copy_backward_result<I1, I2>
//     ranges::copy_backward(I1 first, S1 last, I2 result);
// template<bidirectional_range R, bidirectional_iterator I>
//   requires indirectly_copyable<iterator_t<R>, I>
//   constexpr ranges::copy_backward_result<borrowed_iterator_t<R>, I>
//     ranges::copy_backward(R&& r, I result);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class In, class Out = In, class Sent = sentinel_wrapper<In>>
concept HasCopyBackwardIt = requires(In in, Sent sent, Out out) { std::ranges::copy_backward(in, sent, out); };

static_assert(HasCopyBackwardIt<int*>);
static_assert(!HasCopyBackwardIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCopyBackwardIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCopyBackwardIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCopyBackwardIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyCopyable {};
static_assert(!HasCopyBackwardIt<int*, NotIndirectlyCopyable*>);
static_assert(!HasCopyBackwardIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasCopyBackwardIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range, class Out>
concept HasCopyBackwardR = requires(Range range, Out out) { std::ranges::copy_backward(range, out); };

static_assert(HasCopyBackwardR<std::array<int, 10>, int*>);
static_assert(!HasCopyBackwardR<InputRangeNotDerivedFrom, int*>);
static_assert(!HasCopyBackwardR<InputRangeNotIndirectlyReadable, int*>);
static_assert(!HasCopyBackwardR<InputRangeNotInputOrOutputIterator, int*>);
static_assert(!HasCopyBackwardR<WeaklyIncrementableNotMovable, int*>);
static_assert(!HasCopyBackwardR<UncheckedRange<NotIndirectlyCopyable*>, int*>);
static_assert(!HasCopyBackwardR<InputRangeNotSentinelSemiregular, int*>);
static_assert(!HasCopyBackwardR<InputRangeNotSentinelEqualityComparableWith, int*>);

static_assert(std::is_same_v<std::ranges::copy_result<int, long>, std::ranges::in_out_result<int, long>>);

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  { // simple test
    {
      std::array in {1, 2, 3, 4};
      std::array<int, 4> out;
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
        std::ranges::copy_backward(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data() + out.size()));
      assert(in == out);
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
    {
      std::array in {1, 2, 3, 4};
      std::array<int, 4> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
          std::ranges::copy_backward(range, Out(out.data() + out.size()));
      assert(in == out);
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto ret =
          std::ranges::copy_backward(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data() + out.size()));
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      auto ret = std::ranges::copy_backward(range, Out(out.data()));
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
  }
}

template <class Out>
constexpr void test_in_iterators() {
  test_iterators<bidirectional_iterator<int*>, Out>();
  test_iterators<random_access_iterator<int*>, Out>();
  test_iterators<contiguous_iterator<int*>, Out>();
}

constexpr bool test() {
  test_in_iterators<bidirectional_iterator<int*>>();
  test_in_iterators<random_access_iterator<int*>>();
  test_in_iterators<contiguous_iterator<int*>>();

  { // check that ranges::dangling is returned
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
      std::ranges::copy_backward(std::array {1, 2, 3, 4}, out.data() + out.size());
    assert(ret.out == out.data());
    assert((out == std::array{1, 2, 3, 4}));
  }

  { // check that an iterator is returned with a borrowing range
    std::array in {1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<int*, int*>> auto ret =
        std::ranges::copy_backward(std::views::all(in), out.data() + out.size());
    assert(ret.in == in.data());
    assert(ret.out == out.data());
    assert(in == out);
  }

  { // check that every element is copied exactly once
    struct CopyOnce {
      bool copied = false;
      constexpr CopyOnce() = default;
      constexpr CopyOnce(const CopyOnce& other) = delete;
      constexpr CopyOnce& operator=(const CopyOnce& other) {
        assert(!other.copied);
        copied = true;
        return *this;
      }
    };
    {
      std::array<CopyOnce, 4> in {};
      std::array<CopyOnce, 4> out {};
      auto ret = std::ranges::copy_backward(in.begin(), in.end(), out.end());
      assert(ret.in == in.begin());
      assert(ret.out == out.begin());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
    {
      std::array<CopyOnce, 4> in {};
      std::array<CopyOnce, 4> out {};
      auto ret = std::ranges::copy_backward(in, out.end());
      assert(ret.in == in.begin());
      assert(ret.out == out.begin());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
  }

  { // check that the range is copied backwards
    struct OnlyBackwardsCopyable {
      OnlyBackwardsCopyable* next = nullptr;
      bool canCopy = false;
      OnlyBackwardsCopyable() = default;
      constexpr OnlyBackwardsCopyable& operator=(const OnlyBackwardsCopyable&) {
        assert(canCopy);
        if (next != nullptr)
          next->canCopy = true;
        return *this;
      }
    };
    {
      std::array<OnlyBackwardsCopyable, 3> in {};
      std::array<OnlyBackwardsCopyable, 3> out {};
      out[1].next = &out[0];
      out[2].next = &out[1];
      out[2].canCopy = true;
      auto ret = std::ranges::copy_backward(in, out.end());
      assert(ret.in == in.begin());
      assert(ret.out == out.begin());
      assert(out[0].canCopy);
      assert(out[1].canCopy);
      assert(out[2].canCopy);
    }
    {
      std::array<OnlyBackwardsCopyable, 3> in {};
      std::array<OnlyBackwardsCopyable, 3> out {};
      out[1].next = &out[0];
      out[2].next = &out[1];
      out[2].canCopy = true;
      auto ret = std::ranges::copy_backward(in.begin(), in.end(), out.end());
      assert(ret.in == in.begin());
      assert(ret.out == out.begin());
      assert(out[0].canCopy);
      assert(out[1].canCopy);
      assert(out[2].canCopy);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
