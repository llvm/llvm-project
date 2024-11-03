//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17


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
#include <deque>
#include <ranges>
#include <vector>

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

template <class In, class Out, class Sent>
constexpr void test_iterators() {
  { // simple test
    {
      std::array in {1, 2, 3, 4};
      std::array<int, 4> out;
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
        std::ranges::copy_backward(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data() + out.size()));
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data());
    }
    {
      std::array in {1, 2, 3, 4};
      std::array<int, 4> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
          std::ranges::copy_backward(range, Out(out.data() + out.size()));
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data());
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto ret =
          std::ranges::copy_backward(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data() + out.size()));
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data());
    }
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      auto ret = std::ranges::copy_backward(range, Out(out.data()));
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data());
    }
  }
}

template <class InContainer, class OutContainer, class In, class Out, class Sent = In>
constexpr void test_containers() {
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
      std::ranges::copy_backward(In(in.begin()), Sent(In(in.end())), Out(out.end()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.begin());
  }
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    auto range = std::ranges::subrange(In(in.begin()), Sent(In(in.end())));
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret = std::ranges::copy_backward(range, Out(out.end()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.begin());
  }
}

template <class Iter, class Sent>
constexpr void test_join_view() {
  auto to_subranges = std::views::transform([](auto& vec) {
          return std::ranges::subrange(Iter(vec.begin()), Sent(Iter(vec.end())));
        });

  { // segmented -> contiguous
    std::vector<std::vector<int>> vectors = {};
    auto range = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array<int, 0> arr;

    std::ranges::copy_backward(subrange_vector | std::views::join, arr.end());
    assert(std::ranges::equal(arr, std::array<int, 0>{}));
  }
  { // segmented -> contiguous
    std::vector<std::vector<int>> vectors = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10}, {}};
    auto range = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array<int, 10> arr;

    std::ranges::copy_backward(subrange_vector | std::views::join, arr.end());
    assert(std::ranges::equal(arr, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
  { // contiguous -> segmented
    std::vector<std::vector<int>> vectors = {{0, 0, 0, 0}, {0, 0}, {0, 0, 0, 0}, {}};
    auto range = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::ranges::copy_backward(arr, (subrange_vector | std::views::join).end());
    assert(std::ranges::equal(subrange_vector | std::views::join, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
  { // segmented -> segmented
    std::vector<std::vector<int>> vectors = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10}, {}};
    auto range1 = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range1.begin(), range1.end());
    std::vector<std::vector<int>> to_vectors = {{0, 0, 0, 0}, {0, 0, 0, 0}, {}, {0, 0}};
    auto range2 = to_vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> to_subrange_vector(range2.begin(), range2.end());

    std::ranges::copy_backward(subrange_vector | std::views::join, (to_subrange_vector | std::views::join).end());
    assert(std::ranges::equal(to_subrange_vector | std::views::join, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
}

template <class>
constexpr bool is_proxy_iterator = false;

template <class Iter>
constexpr bool is_proxy_iterator<ProxyIterator<Iter>> = true;

template <template <class> class InIter, template <class> class OutIter>
constexpr void test_sentinels() {
  test_iterators<InIter<int*>, OutIter<int*>, InIter<int*>>();
  test_iterators<InIter<int*>, OutIter<int*>, sentinel_wrapper<InIter<int*>>>();
  test_iterators<InIter<int*>, OutIter<int*>, sized_sentinel<InIter<int*>>>();

  if constexpr (!std::is_same_v<InIter<int*>, contiguous_iterator<int*>> &&
                !std::is_same_v<OutIter<int*>, contiguous_iterator<int*>> &&
                !std::is_same_v<InIter<int*>, ContiguousProxyIterator<int*>> &&
                !std::is_same_v<OutIter<int*>, ContiguousProxyIterator<int*>>) {
    if (!std::is_constant_evaluated()) {
      test_containers<std::deque<int>,
                      std::deque<int>,
                      InIter<std::deque<int>::iterator>,
                      OutIter<std::deque<int>::iterator>>();
      test_containers<std::deque<int>,
                      std::vector<int>,
                      InIter<std::deque<int>::iterator>,
                      OutIter<std::vector<int>::iterator>>();
      test_containers<std::vector<int>,
                      std::deque<int>,
                      InIter<std::vector<int>::iterator>,
                      OutIter<std::deque<int>::iterator>>();
      test_containers<std::vector<int>,
                      std::vector<int>,
                      InIter<std::vector<int>::iterator>,
                      OutIter<std::vector<int>::iterator>>();
    }
    if constexpr (!is_proxy_iterator<InIter<int*>>)
      test_join_view<InIter<std::vector<int>::iterator>, InIter<std::vector<int>::iterator>>();
  }
}

template <template <class> class Out>
constexpr void test_in_iterators() {
  test_sentinels<bidirectional_iterator, Out>();
  test_sentinels<random_access_iterator, Out>();
  test_sentinels<contiguous_iterator, Out>();
  test_sentinels<std::type_identity_t, Out>();
}

template <template <class> class Out>
constexpr void test_proxy_in_iterators() {
  test_sentinels<BidirectionalProxyIterator, Out>();
  test_sentinels<RandomAccessProxyIterator, Out>();
  test_sentinels<ContiguousProxyIterator, Out>();
  test_sentinels<ProxyIterator, Out>();
}

constexpr bool test() {
  test_in_iterators<bidirectional_iterator>();
  test_in_iterators<random_access_iterator>();
  test_in_iterators<contiguous_iterator>();
  test_in_iterators<std::type_identity_t>();

  test_proxy_in_iterators<BidirectionalProxyIterator>();
  test_proxy_in_iterators<RandomAccessProxyIterator>();
  test_proxy_in_iterators<ContiguousProxyIterator>();

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
    assert(ret.in == in.data() + in.size());
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
      assert(ret.in == in.end());
      assert(ret.out == out.begin());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
    }
    {
      std::array<CopyOnce, 4> in {};
      std::array<CopyOnce, 4> out {};
      auto ret = std::ranges::copy_backward(in, out.end());
      assert(ret.in == in.end());
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
      assert(ret.in == in.end());
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
      assert(ret.in == in.end());
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
