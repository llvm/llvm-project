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

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::copy_result<I, O> ranges::copy(I first, S last, O result);
// template<input_range R, weakly_incrementable O>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr ranges::copy_result<borrowed_iterator_t<R>, O> ranges::copy(R&& r, O result);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

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

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  { // simple test
    {
      std::array in {1, 2, 3, 4};
      std::array<int, 4> out;
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
        std::ranges::copy(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data()));
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
    {
      std::array in {1, 2, 3, 4};
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
      auto ret = std::ranges::copy(range, Out(out.data()));
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
  }
}

template <class In, class Out>
constexpr void test_sentinels() {
  test_iterators<In, Out>();
  test_iterators<In, Out, sized_sentinel<In>>();
  test_iterators<In, Out, sentinel_wrapper<In>>();
}

template <class Out>
constexpr void test_in_iterators() {
  test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_sentinels<forward_iterator<int*>, Out>();
  test_sentinels<bidirectional_iterator<int*>, Out>();
  test_sentinels<random_access_iterator<int*>, Out>();
  test_sentinels<contiguous_iterator<int*>, Out>();
}

template <class Out>
constexpr void test_proxy_in_iterators() {
  test_iterators<ProxyIterator<cpp20_input_iterator<int*>>, Out, sentinel_wrapper<ProxyIterator<cpp20_input_iterator<int*>>>>();
  test_iterators<ProxyIterator<forward_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<bidirectional_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<random_access_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<contiguous_iterator<int*>>, Out>();
}

constexpr bool test() {
  test_in_iterators<cpp20_input_iterator<int*>>();
  test_in_iterators<forward_iterator<int*>>();
  test_in_iterators<bidirectional_iterator<int*>>();
  test_in_iterators<random_access_iterator<int*>>();
  test_in_iterators<contiguous_iterator<int*>>();

  test_proxy_in_iterators<ProxyIterator<cpp20_input_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<forward_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<bidirectional_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<random_access_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<contiguous_iterator<int*>>>();

  { // check that ranges::dangling is returned
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
      std::ranges::copy(std::array {1, 2, 3, 4}, out.data());
    assert(ret.out == out.data() + 4);
    assert((out == std::array{1, 2, 3, 4}));
  }

  { // check that an iterator is returned with a borrowing range
    std::array in {1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<int*, int*>> auto ret = std::ranges::copy(std::views::all(in), out.data());
    assert(ret.in == in.data() + 4);
    assert(ret.out == out.data() + 4);
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
      auto ret = std::ranges::copy(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
    {
      std::array<CopyOnce, 4> in {};
      std::array<CopyOnce, 4> out {};
      auto ret = std::ranges::copy(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
  }

  { // check that the range is copied forwards
    struct OnlyForwardsCopyable {
      OnlyForwardsCopyable* next = nullptr;
      bool canCopy = false;
      OnlyForwardsCopyable() = default;
      constexpr OnlyForwardsCopyable& operator=(const OnlyForwardsCopyable&) {
        assert(canCopy);
        if (next != nullptr)
          next->canCopy = true;
        return *this;
      }
    };
    {
      std::array<OnlyForwardsCopyable, 3> in {};
      std::array<OnlyForwardsCopyable, 3> out {};
      out[0].next = &out[1];
      out[1].next = &out[2];
      out[0].canCopy = true;
      auto ret = std::ranges::copy(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].canCopy);
      assert(out[1].canCopy);
      assert(out[2].canCopy);
    }
    {
      std::array<OnlyForwardsCopyable, 3> in {};
      std::array<OnlyForwardsCopyable, 3> out {};
      out[0].next = &out[1];
      out[1].next = &out[2];
      out[0].canCopy = true;
      auto ret = std::ranges::copy(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
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
