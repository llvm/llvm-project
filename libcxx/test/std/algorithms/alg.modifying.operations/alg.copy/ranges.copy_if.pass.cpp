//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::copy_if_result<I, O>
//     ranges::copy_if(I first, S last, O result, Pred pred, Proj proj = {});
// template<input_range R, weakly_incrementable O, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr ranges::copy_if_result<borrowed_iterator_t<R>, O>
//     ranges::copy_if(R&& r, O result, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Functor {
  bool operator()(int);
};

template <class In, class Out = In, class Sent = sentinel_wrapper<In>, class Func = Functor>
concept HasCopyIfIt = requires(In first, Sent last, Out result) { std::ranges::copy_if(first, last, result, Func{}); };

static_assert(HasCopyIfIt<int*>);
static_assert(!HasCopyIfIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCopyIfIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCopyIfIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCopyIfIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyCopyable {};
static_assert(!HasCopyIfIt<int*, NotIndirectlyCopyable*>);
static_assert(!HasCopyIfIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasCopyIfIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

static_assert(!HasCopyIfIt<int*, int*, int*, IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasCopyIfIt<int*, int*, int*, IndirectUnaryPredicateNotPredicate>);

template <class Range, class Out, class Func = Functor>
concept HasCopyIfR = requires(Range range, Out out) { std::ranges::copy_if(range, out, Func{}); };

static_assert(HasCopyIfR<std::array<int, 10>, int*>);
static_assert(!HasCopyIfR<InputRangeNotDerivedFrom, int*>);
static_assert(!HasCopyIfR<InputRangeNotIndirectlyReadable, int*>);
static_assert(!HasCopyIfR<InputRangeNotInputOrOutputIterator, int*>);
static_assert(!HasCopyIfR<WeaklyIncrementableNotMovable, int*>);
static_assert(!HasCopyIfR<UncheckedRange<NotIndirectlyCopyable*>, int*>);
static_assert(!HasCopyIfR<InputRangeNotSentinelSemiregular, int*>);
static_assert(!HasCopyIfR<InputRangeNotSentinelEqualityComparableWith, int*>);

static_assert(std::is_same_v<std::ranges::copy_if_result<int, long>, std::ranges::in_out_result<int, long>>);

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  { // simple test
    {
      std::array in = {1, 2, 3, 4};
      std::array<int, 4> out;
      std::same_as<std::ranges::copy_if_result<In, Out>> auto ret =
          std::ranges::copy_if(In(in.data()),
                               Sent(In(in.data() + in.size())),
                               Out(out.data()),
                               [](int) { return true; });
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
    {
      std::array in = {1, 2, 3, 4};
      std::array<int, 4> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      std::same_as<std::ranges::copy_if_result<In, Out>> auto ret =
          std::ranges::copy_if(range, Out(out.data()), [](int) { return true; });
      assert(in == out);
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto ret = std::ranges::copy_if(In(in.data()), Sent(In(in.data())), Out(out.data()), [](int) { return true; });
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
    {
      std::array<int, 0> in;
      std::array<int, 0> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data())));
      auto ret = std::ranges::copy_if(range, Out(out.data()), [](int) { return true; });
      assert(base(ret.in) == in.data());
      assert(base(ret.out) == out.data());
    }
  }

  { // check that the predicate is used
    {
      std::array in = {4, 6, 87, 3, 88, 44, 45, 9};
      std::array<int, 4> out;
      auto ret = std::ranges::copy_if(In(in.data()),
                                      Sent(In(in.data() + in.size())),
                                      Out(out.data()),
                                      [](int i) { return i % 2 == 0; });
      assert((out == std::array{4, 6, 88, 44}));
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
    {
      std::array in = {4, 6, 87, 3, 88, 44, 45, 9};
      std::array<int, 4> out;
      auto range = std::ranges::subrange(In(in.data()), Sent(In(in.data() + in.size())));
      auto ret = std::ranges::copy_if(range, Out(out.data()), [](int i) { return i % 2 == 0; });
      assert((out == std::array{4, 6, 88, 44}));
      assert(base(ret.in) == in.data() + in.size());
      assert(base(ret.out) == out.data() + out.size());
    }
  }
}

template <class Out>
constexpr bool test_in_iterators() {
  test_iterators<cpp17_input_iterator<int*>, Out, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>, Out>();
  test_iterators<bidirectional_iterator<int*>, Out>();
  test_iterators<random_access_iterator<int*>, Out>();
  test_iterators<contiguous_iterator<int*>, Out>();
  test_iterators<int*, Out>();

  return true;
}

constexpr bool test() {
  test_in_iterators<cpp17_output_iterator<int*>>();
  test_in_iterators<cpp20_output_iterator<int*>>();
  test_in_iterators<forward_iterator<int*>>();
  test_in_iterators<bidirectional_iterator<int*>>();
  test_in_iterators<random_access_iterator<int*>>();
  test_in_iterators<contiguous_iterator<int*>>();
  test_in_iterators<int*>();

  { // check that std::invoke is used
    {
      struct S { int val; int other; };
      std::array<S, 4> in = {{{4, 2}, {1, 3}, {3, 4}, {3, 5}}};
      std::array<S, 2> out;
      auto ret = std::ranges::copy_if(in.begin(), in.end(), out.begin(), [](int i) { return i == 3; }, &S::val);
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].val == 3);
      assert(out[0].other == 4);
      assert(out[1].val == 3);
      assert(out[1].other == 5);
    }
    {
      struct S { int val; int other; };
      std::array<S, 4> in = {{{4, 2}, {1, 3}, {3, 4}, {3, 5}}};
      std::array<S, 2> out;
      auto ret = std::ranges::copy_if(in, out.begin(), [](int i) { return i == 3; }, &S::val);
      assert(ret.in == in.end());
      assert(ret.out == out.end());
      assert(out[0].val == 3);
      assert(out[0].other == 4);
      assert(out[1].val == 3);
      assert(out[1].other == 5);
    }
  }

  { // check that the complexity requirements are met
    {
      int predicateCount = 0;
      int projectionCount = 0;
      auto pred = [&](int i) { ++predicateCount; return i != 0; };
      auto proj = [&](int i) { ++projectionCount; return i; };

      int a[] = {5, 4, 3, 2, 1};
      int b[5];
      std::ranges::copy_if(a, a + 5, b, pred, proj);
      assert(predicateCount == 5);
      assert(projectionCount == 5);
    }
    {
      int predicateCount = 0;
      int projectionCount = 0;
      auto pred = [&](int i) { ++predicateCount; return i != 0; };
      auto proj = [&](int i) { ++projectionCount; return i; };

      int a[] = {5, 4, 3, 2, 1};
      int b[5];
      std::ranges::copy_if(a, b, pred, proj);
      assert(predicateCount == 5);
      assert(projectionCount == 5);
    }
  }

  { // test proxy iterator
    {
      std::array in = {4, 6, 87, 3, 88, 44, 45, 9};
      std::array<int, 4> out;

      ProxyRange proxyIn{in};
      ProxyRange proxyOut{out};

      std::ranges::copy_if(proxyIn.begin(), proxyIn.end(), proxyOut.begin(),
                          [](auto const& i) { return i.data % 2 == 0; });
      assert((out == std::array{4, 6, 88, 44}));
    }
    {
      std::array in = {4, 6, 87, 3, 88, 44, 45, 9};
      std::array<int, 4> out;

      ProxyRange proxyIn{in};
      ProxyRange proxyOut{out};

      std::ranges::copy_if(proxyIn, proxyOut.begin(), [](const auto& i) { return i.data % 2 == 0; });
      assert((out == std::array{4, 6, 88, 44}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
