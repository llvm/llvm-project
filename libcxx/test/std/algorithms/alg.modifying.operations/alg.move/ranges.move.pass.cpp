//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O>
//   requires indirectly_movable<I, O>
//   constexpr ranges::move_result<I, O>
//     ranges::move(I first, S last, O result);
// template<input_range R, weakly_incrementable O>
//   requires indirectly_movable<iterator_t<R>, O>
//   constexpr ranges::move_result<borrowed_iterator_t<R>, O>
//     ranges::move(R&& r, O result);

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <iterator>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"

template <class In, class Out = In, class Sent = sentinel_wrapper<In>>
concept HasMoveIt = requires(In in, Sent sent, Out out) { std::ranges::move(in, sent, out); };

static_assert(HasMoveIt<int*>);
static_assert(!HasMoveIt<InputIteratorNotDerivedFrom>);
static_assert(!HasMoveIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasMoveIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasMoveIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyMovable {};
static_assert(!HasMoveIt<int*, NotIndirectlyMovable*>);
static_assert(!HasMoveIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasMoveIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range, class Out>
concept HasMoveR = requires(Range range, Out out) { std::ranges::move(range, out); };

static_assert(HasMoveR<std::array<int, 10>, int*>);
static_assert(!HasMoveR<InputRangeNotDerivedFrom, int*>);
static_assert(!HasMoveR<InputRangeNotIndirectlyReadable, int*>);
static_assert(!HasMoveR<InputRangeNotInputOrOutputIterator, int*>);
static_assert(!HasMoveR<WeaklyIncrementableNotMovable, int*>);
static_assert(!HasMoveR<UncheckedRange<NotIndirectlyMovable*>, int*>);
static_assert(!HasMoveR<InputRangeNotSentinelSemiregular, int*>);
static_assert(!HasMoveR<InputRangeNotSentinelEqualityComparableWith, int*>);
static_assert(!HasMoveR<UncheckedRange<int*>, WeaklyIncrementableNotMovable>);

static_assert(std::is_same_v<std::ranges::move_result<int, long>, std::ranges::in_out_result<int, long>>);

template <class In, class Out, class Sent, int N>
constexpr void test(std::array<int, N> in) {
  {
    std::array<int, N> out;
    std::same_as<std::ranges::in_out_result<In, Out>> decltype(auto) ret =
      std::ranges::move(In(in.data()), Sent(In(in.data() + in.size())), Out(out.data()));
    assert(in == out);
    assert(base(ret.in) == in.data() + in.size());
    assert(base(ret.out) == out.data() + out.size());
  }
  {
    std::array<int, N> out;
    auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
    std::same_as<std::ranges::in_out_result<In, Out>> decltype(auto) ret =
        std::ranges::move(range, Out(out.data()));
    assert(in == out);
    assert(base(ret.in) == in.data() + in.size());
    assert(base(ret.out) == out.data() + out.size());
  }
}

template <class InContainer, class OutContainer, class In, class Out, class Sent = In>
constexpr void test_containers() {
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
      std::ranges::move(In(in.begin()), Sent(In(in.end())), Out(out.begin()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.end());
  }
  {
    InContainer in {1, 2, 3, 4};
    OutContainer out(4);
    auto range = std::ranges::subrange(In(in.begin()), Sent(In(in.end())));
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret = std::ranges::move(range, Out(out.begin()));
    assert(std::ranges::equal(in, out));
    assert(base(ret.in) == in.end());
    assert(base(ret.out) == out.end());
  }
}

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  // simple test
  test<In, Out, Sent, 4>({1, 2, 3, 4});
  // check that an empty range works
  test<In, Out, Sent, 0>({});
}

template <template <class> class In, template <class> class Out>
constexpr void test_sentinels() {
  test_iterators<In<int*>, Out<int*>>();
  test_iterators<In<int*>, Out<int*>, sized_sentinel<In<int*>>>();
  test_iterators<In<int*>, Out<int*>, sentinel_wrapper<In<int*>>>();

  if (!std::is_constant_evaluated()) {
    if constexpr (!std::is_same_v<In<int*>, contiguous_iterator<int*>> &&
                  !std::is_same_v<Out<int*>, contiguous_iterator<int*>> &&
                  !std::is_same_v<In<int*>, ContiguousProxyIterator<int*>> &&
                  !std::is_same_v<Out<int*>, ContiguousProxyIterator<int*>>) {
      test_containers<std::deque<int>,
                      std::deque<int>,
                      In<std::deque<int>::iterator>,
                      Out<std::deque<int>::iterator>>();
      test_containers<std::deque<int>,
                      std::vector<int>,
                      In<std::deque<int>::iterator>,
                      Out<std::vector<int>::iterator>>();
      test_containers<std::vector<int>,
                      std::deque<int>,
                      In<std::vector<int>::iterator>,
                      Out<std::deque<int>::iterator>>();
      test_containers<std::vector<int>,
                      std::vector<int>,
                      In<std::vector<int>::iterator>,
                      Out<std::vector<int>::iterator>>();
    }
  }
}

template <template <class> class Out>
constexpr void test_in_iterators() {
  test_iterators<cpp20_input_iterator<int*>, Out<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_sentinels<forward_iterator, Out>();
  test_sentinels<bidirectional_iterator, Out>();
  test_sentinels<random_access_iterator, Out>();
  test_sentinels<contiguous_iterator, Out>();
  test_sentinels<std::type_identity_t, Out>();
}

template <template <class> class Out>
constexpr void test_proxy_in_iterators() {
  test_iterators<ProxyIterator<cpp20_input_iterator<int*>>,
                 Out<int*>,
                 sentinel_wrapper<ProxyIterator<cpp20_input_iterator<int*>>>>();
  test_sentinels<ForwardProxyIterator, Out>();
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

  friend constexpr int iter_move(const IteratorWithMoveIter&) { return 42; }

  constexpr bool operator==(const IteratorWithMoveIter& other) const = default;
};

// cpp17_intput_iterator has a defaulted template argument
template <class Iter>
using Cpp17InIter = cpp17_input_iterator<Iter>;

constexpr bool test() {
  test_in_iterators<cpp17_output_iterator>();
  test_in_iterators<cpp20_output_iterator>();
  test_in_iterators<Cpp17InIter>();
  test_in_iterators<cpp20_input_iterator>();
  test_in_iterators<forward_iterator>();
  test_in_iterators<bidirectional_iterator>();
  test_in_iterators<random_access_iterator>();
  test_in_iterators<contiguous_iterator>();
  test_in_iterators<std::type_identity_t>();

  test_proxy_in_iterators<Cpp20InputProxyIterator>();
  test_proxy_in_iterators<ForwardProxyIterator>();
  test_proxy_in_iterators<BidirectionalProxyIterator>();
  test_proxy_in_iterators<RandomAccessProxyIterator>();
  test_proxy_in_iterators<ContiguousProxyIterator>();
  test_proxy_in_iterators<ProxyIterator>();

  { // check that a move-only type works
    // When non-trivial
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      std::ranges::move(a, std::begin(b));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      std::ranges::move(std::begin(a), std::end(a), std::begin(b));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }

    // When trivial
    {
      TrivialMoveOnly a[] = {1, 2, 3};
      TrivialMoveOnly b[3];
      std::ranges::move(a, std::begin(b));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
    {
      TrivialMoveOnly a[] = {1, 2, 3};
      TrivialMoveOnly b[3];
      std::ranges::move(std::begin(a), std::end(a), std::begin(b));
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
      std::ranges::move(proxyA, std::begin(proxyB));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
    {
      MoveOnly a[] = {1, 2, 3};
      MoveOnly b[3];
      ProxyRange proxyA{a};
      ProxyRange proxyB{b};
      std::ranges::move(std::begin(proxyA), std::end(proxyA), std::begin(proxyB));
      assert(b[0].get() == 1);
      assert(b[1].get() == 2);
      assert(b[2].get() == 3);
    }
  }

  { // check that ranges::dangling is returned
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> decltype(auto) ret =
      std::ranges::move(std::array {1, 2, 3, 4}, out.data());
    assert(ret.out == out.data() + 4);
    assert((out == std::array{1, 2, 3, 4}));
  }

  { // check that an iterator is returned with a borrowing range
    std::array in {1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<int*, int*>> decltype(auto) ret =
        std::ranges::move(std::views::all(in), out.data());
    assert(ret.in == in.data() + 4);
    assert(ret.out == out.data() + 4);
    assert(in == out);
  }

  { // check that every element is moved exactly once
    struct MoveOnce {
      bool moved = false;
      constexpr MoveOnce() = default;
      constexpr MoveOnce(const MoveOnce& other) = delete;
      constexpr MoveOnce& operator=(MoveOnce&& other) {
        assert(!other.moved);
        moved = true;
        return *this;
      }
    };
    {
      std::array<MoveOnce, 4> in {};
      std::array<MoveOnce, 4> out {};
      auto ret = std::ranges::move(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.moved; }));
    }
    {
      std::array<MoveOnce, 4> in {};
      std::array<MoveOnce, 4> out {};
      auto ret = std::ranges::move(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.moved; }));
    }
  }

  { // check that the range is moved forwards
    struct OnlyForwardsMovable {
      OnlyForwardsMovable* next = nullptr;
      bool canMove = false;
      OnlyForwardsMovable() = default;
      constexpr OnlyForwardsMovable& operator=(OnlyForwardsMovable&&) {
        assert(canMove);
        if (next != nullptr)
          next->canMove = true;
        return *this;
      }
    };
    {
      std::array<OnlyForwardsMovable, 3> in {};
      std::array<OnlyForwardsMovable, 3> out {};
      out[0].next = &out[1];
      out[1].next = &out[2];
      out[0].canMove = true;
      auto ret = std::ranges::move(in.begin(), in.end(), out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].canMove);
      assert(out[1].canMove);
      assert(out[2].canMove);
    }
    {
      std::array<OnlyForwardsMovable, 3> in {};
      std::array<OnlyForwardsMovable, 3> out {};
      out[0].next = &out[1];
      out[1].next = &out[2];
      out[0].canMove = true;
      auto ret = std::ranges::move(in, out.begin());
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].canMove);
      assert(out[1].canMove);
      assert(out[2].canMove);
    }
  }

  { // check that iter_move is used properly
    {
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> b;
      auto ret = std::ranges::move(IteratorWithMoveIter(a), IteratorWithMoveIter(a + 4), b.data());
      assert(ret.in == a + 4);
      assert(ret.out == b.data() + 4);
      assert((b == std::array {42, 42, 42, 42}));
    }
    {
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> b;
      auto range = std::ranges::subrange(IteratorWithMoveIter(a), IteratorWithMoveIter(a + 4));
      auto ret = std::ranges::move(range, b.data());
      assert(ret.in == a + 4);
      assert(ret.out == b.data() + 4);
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
