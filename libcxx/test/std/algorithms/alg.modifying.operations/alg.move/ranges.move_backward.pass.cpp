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
//   requires indirectly_movable<I1, I2>
//   constexpr ranges::move_backward_result<I1, I2>
//     ranges::move_backward(I1 first, S1 last, I2 result);
// template<bidirectional_range R, bidirectional_iterator I>
//   requires indirectly_movable<iterator_t<R>, I>
//   constexpr ranges::move_backward_result<borrowed_iterator_t<R>, I>
//     ranges::move_backward(R&& r, I result);

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"

template <class In, class Out = In, class Sent = sentinel_wrapper<In>>
concept HasMoveBackwardIt = requires(In in, Sent sent, Out out) { std::ranges::move_backward(in, sent, out); };

static_assert(HasMoveBackwardIt<int*>);
static_assert(!HasMoveBackwardIt<InputIteratorNotDerivedFrom>);
static_assert(!HasMoveBackwardIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasMoveBackwardIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasMoveBackwardIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyCopyable {};
static_assert(!HasMoveBackwardIt<int*, NotIndirectlyCopyable*>);
static_assert(!HasMoveBackwardIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasMoveBackwardIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range, class Out>
concept HasMoveBackwardR = requires(Range range, Out out) { std::ranges::move_backward(range, out); };

static_assert(HasMoveBackwardR<std::array<int, 10>, int*>);
static_assert(!HasMoveBackwardR<InputRangeNotDerivedFrom, int*>);
static_assert(!HasMoveBackwardR<InputRangeNotIndirectlyReadable, int*>);
static_assert(!HasMoveBackwardR<InputRangeNotInputOrOutputIterator, int*>);
static_assert(!HasMoveBackwardR<WeaklyIncrementableNotMovable, int*>);
static_assert(!HasMoveBackwardR<UncheckedRange<NotIndirectlyCopyable*>, int*>);
static_assert(!HasMoveBackwardR<InputRangeNotSentinelSemiregular, int*>);
static_assert(!HasMoveBackwardR<InputRangeNotSentinelEqualityComparableWith, int*>);
static_assert(!HasMoveBackwardR<UncheckedRange<int*>, WeaklyIncrementableNotMovable>);

static_assert(std::is_same_v<std::ranges::copy_result<int, long>, std::ranges::in_out_result<int, long>>);

template <class In, class Out, class Sent, int N>
constexpr void test(std::array<int, N> in) {
  {
    std::array<int, N> out;
    std::same_as<std::ranges::in_out_result<In, Out>> decltype(auto) ret =
      std::ranges::move_backward(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data() + out.size()));
    assert(in == out);
    assert(base(ret.in) == in.data() + in.size());
    assert(base(ret.out) == out.data());
  }
  {
    std::array<int, N> out;
    auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
    std::same_as<std::ranges::in_out_result<In, Out>> decltype(auto) ret =
        std::ranges::move_backward(range, Out(out.data() + out.size()));
    assert(in == out);
    assert(base(ret.in) == in.data() + in.size());
    assert(base(ret.out) == out.data());
  }
}

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  // simple test
  test<In, Out, Sent, 4>({1, 2, 3, 4});
  // check that an empty range works
  test<In, Out, Sent, 0>({});
}

template <class InContainer, class OutContainer, class In, class Out, class Sent = In>
constexpr void test_containers() {
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
      std::ranges::move_backward(In(in.begin()), Sent(In(in.end())), Out(out.end()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.begin());
  }
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    auto range = std::ranges::subrange(In(in.begin()), Sent(In(in.end())));
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret = std::ranges::move_backward(range, Out(out.end()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.begin());
  }
}

template <template <class> class InIter, template <class> class OutIter>
constexpr void test_sentinels() {
  test_iterators<InIter<int*>, OutIter<int*>, InIter<int*>>();
  test_iterators<InIter<int*>, OutIter<int*>, sentinel_wrapper<InIter<int*>>>();
  test_iterators<InIter<int*>, OutIter<int*>, sized_sentinel<InIter<int*>>>();

  if (!std::is_constant_evaluated()) {
    if constexpr (!std::is_same_v<InIter<int*>, contiguous_iterator<int*>> &&
                  !std::is_same_v<OutIter<int*>, contiguous_iterator<int*>> &&
                  !std::is_same_v<InIter<int*>, ContiguousProxyIterator<int*>> &&
                  !std::is_same_v<OutIter<int*>, ContiguousProxyIterator<int*>>) {
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

struct IteratorWithMoveIter {
  using value_type = int;
  using difference_type = int;
  explicit IteratorWithMoveIter() = default;
  int* ptr;
  constexpr IteratorWithMoveIter(int* ptr_) : ptr(ptr_) {}

  constexpr int& operator*() const; // iterator with iter_move should not be dereferenced

  constexpr IteratorWithMoveIter& operator++() { ++ptr; return *this; }
  constexpr IteratorWithMoveIter operator++(int) { auto ret = *this; ++*this; return ret; }

  constexpr IteratorWithMoveIter& operator--() { --ptr; return *this; }
  constexpr IteratorWithMoveIter operator--(int) { auto ret = *this; --*this; return ret; }

  friend constexpr int iter_move(const IteratorWithMoveIter&) { return 42; }

  constexpr bool operator==(const IteratorWithMoveIter& other) const = default;
};

constexpr bool test() {
  test_in_iterators<bidirectional_iterator>();
  test_in_iterators<random_access_iterator>();
  test_in_iterators<contiguous_iterator>();
  test_in_iterators<std::type_identity_t>();

  test_proxy_in_iterators<BidirectionalProxyIterator>();
  test_proxy_in_iterators<RandomAccessProxyIterator>();
  test_proxy_in_iterators<ContiguousProxyIterator>();
  test_proxy_in_iterators<ProxyIterator>();

  { // check that a move-only type works
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      std::ranges::move_backward(a, std::end(b));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      std::ranges::move_backward(std::begin(a), std::end(a), std::end(b));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
  }

  { // check that a move-only type works for ProxyIterator
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      ProxyRange proxyA{a};
      ProxyRange proxyB{b};
      std::ranges::move_backward(proxyA, std::ranges::next(proxyB.begin(), std::end(proxyB)));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      ProxyRange proxyA{a};
      ProxyRange proxyB{b};
      std::ranges::move_backward(std::begin(proxyA), std::end(proxyA),  std::ranges::next(proxyB.begin(), std::end(proxyB)));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
  }

  { // check that ranges::dangling is returned
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
      std::ranges::move_backward(std::array {1, 2, 3, 4}, out.data() + out.size());
    assert(ret.out == out.data());
    assert((out == std::array{1, 2, 3, 4}));
  }

  { // check that an iterator is returned with a borrowing range
    std::array in {1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<int*, int*>> auto ret =
        std::ranges::move_backward(std::views::all(in), out.data() + out.size());
    assert(ret.in == in.data() + in.size());
    assert(ret.out == out.data());
    assert(in == out);
  }

  { // check that every element is moved exactly once
    struct MoveOnce {
      bool moved = false;
      constexpr MoveOnce() = default;
      constexpr MoveOnce(const MoveOnce& other) = delete;
      constexpr MoveOnce& operator=(const MoveOnce& other) {
        assert(!other.moved);
        moved = true;
        return *this;
      }
    };
    {
      std::array<MoveOnce, 4> in {};
      std::array<MoveOnce, 4> out {};
      auto ret = std::ranges::move_backward(in.begin(), in.end(), out.end());
      assert(ret.in == in.end());
      assert(ret.out == out.begin());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.moved; }));
    }
    {
      std::array<MoveOnce, 4> in {};
      std::array<MoveOnce, 4> out {};
      auto ret = std::ranges::move_backward(in, out.end());
      assert(ret.in == in.end());
      assert(ret.out == out.begin());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.moved; }));
    }
  }

  { // check that the range is moved backwards
    struct OnlyBackwardsMovable {
      OnlyBackwardsMovable* next = nullptr;
      bool canMove = false;
      OnlyBackwardsMovable() = default;
      constexpr OnlyBackwardsMovable& operator=(const OnlyBackwardsMovable&) {
        assert(canMove);
        if (next != nullptr)
          next->canMove = true;
        return *this;
      }
    };
    {
      std::array<OnlyBackwardsMovable, 3> in {};
      std::array<OnlyBackwardsMovable, 3> out {};
      out[1].next = &out[0];
      out[2].next = &out[1];
      out[2].canMove = true;
      auto ret = std::ranges::move_backward(in, out.end());
      assert(ret.in == in.end());
      assert(ret.out == out.begin());
      assert(out[0].canMove);
      assert(out[1].canMove);
      assert(out[2].canMove);
    }
    {
      std::array<OnlyBackwardsMovable, 3> in {};
      std::array<OnlyBackwardsMovable, 3> out {};
      out[1].next = &out[0];
      out[2].next = &out[1];
      out[2].canMove = true;
      auto ret = std::ranges::move_backward(in.begin(), in.end(), out.end());
      assert(ret.in == in.end());
      assert(ret.out == out.begin());
      assert(out[0].canMove);
      assert(out[1].canMove);
      assert(out[2].canMove);
    }
  }

  { // check that iter_move is used properly
    {
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> b;
      auto ret = std::ranges::move_backward(IteratorWithMoveIter(a), IteratorWithMoveIter(a + 4), b.data() + b.size());
      assert(ret.in == a + 4);
      assert(ret.out == b.data());
      assert((b == std::array {42, 42, 42, 42}));
    }
    {
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> b;
      auto range = std::ranges::subrange(IteratorWithMoveIter(a), IteratorWithMoveIter(a + 4));
      auto ret = std::ranges::move_backward(range, b.data() + b.size());
      assert(ret.in == a + 4);
      assert(ret.out == b.data());
      assert((b == std::array {42, 42, 42, 42}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
