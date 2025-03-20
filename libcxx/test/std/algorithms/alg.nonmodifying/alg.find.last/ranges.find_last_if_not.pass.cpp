//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr subrange<I> ranges::find_last_if_not(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_subrange_t<R> ranges::find_last_if_not(R&& r, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Predicate {
  bool operator()(int);
};

template <class It, class Sent = It>
concept HasFindLastIfIt = requires(It it, Sent sent) { std::ranges::find_last_if_not(it, sent, Predicate{}); };
static_assert(HasFindLastIfIt<int*>);
static_assert(HasFindLastIfIt<forward_iterator<int*>>);
static_assert(!HasFindLastIfIt<cpp20_input_iterator<int*>>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIfIt<forward_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindLastIfIt<forward_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindLastIfIt<int*, int>);
static_assert(!HasFindLastIfIt<int, int*>);

template <class Pred>
concept HasFindLastIfPred = requires(int* it, Pred pred) { std::ranges::find_last_if_not(it, it, pred); };

static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotPredicate>);

template <class R>
concept HasFindLastIfR = requires(R r) { std::ranges::find_last_if_not(r, Predicate{}); };
static_assert(HasFindLastIfR<std::array<int, 0>>);
static_assert(!HasFindLastIfR<int>);
static_assert(!HasFindLastIfR<ForwardRangeNotDerivedFrom>);
static_assert(!HasFindLastIfR<ForwardRangeNotIncrementable>);
static_assert(!HasFindLastIfR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasFindLastIfR<ForwardRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent>
constexpr auto make_range(auto& a) {
  return std::ranges::subrange(It(std::ranges::begin(a)), Sent(It(std::ranges::end(a))));
}

template <template <class> class IteratorT, template <class> class SentinelT>
constexpr void test_iterator_classes() {
  {
    using it   = IteratorT<int*>;
    using sent = SentinelT<it>;

    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ranges::subrange<it>> auto ret =
          std::ranges::find_last_if_not(it(a), sent(it(a + 4)), [](int x) { return x != 4; });
      assert(base(ret.begin()) == a + 3);
      assert(*ret.begin() == 4);
    }
    {
      int a[] = {1, 2, 3, 4};

      std::same_as<std::ranges::subrange<it>> auto ret =
          std::ranges::find_last_if_not(make_range<it, sent>(a), [](int x) { return x != 4; });
      assert(ret.begin() == it(a + 3));
      assert(*ret.begin() == 4);
    }
  }

  { // check that an empty range works
    using it   = IteratorT<std::ranges::iterator_t<std::array<int, 0>&>>;
    using sent = SentinelT<it>;

    {
      std::array<int, 0> a = {};

      auto ret = std::ranges::find_last_if_not(it(a.begin()), sent(it(a.end())), [](auto&&) { return false; }).begin();
      assert(ret == it(a.end()));
    }
    {
      std::array<int, 0> a = {};

      auto ret = std::ranges::find_last_if_not(make_range<it, sent>(a), [](auto&&) { return false; }).begin();
      assert(ret == it(a.end()));
    }
  }

  { // check that last is returned with no match
    using it   = IteratorT<int*>;
    using sent = SentinelT<it>;

    {
      int a[] = {1, 1, 1};

      auto ret = std::ranges::find_last_if_not(it(a), sent(it(a + 3)), [](auto&&) { return true; }).begin();
      assert(ret == it(a + 3));
    }
    {
      int a[] = {1, 1, 1};

      auto ret = std::ranges::find_last_if_not(make_range<it, sent>(a), [](auto&&) { return true; }).begin();
      assert(ret == it(a + 3));
    }
  }

  { // check that the last element is returned
    struct S {
      int comp;
      int other;
    };
    using it   = IteratorT<S*>;
    using sent = SentinelT<it>;

    S a[] = {{0, 0}, {0, 2}, {0, 1}};

    auto ret = std::ranges::find_last_if_not(
                   it(std::begin(a)), sent(it(std::end(a))), [](int c) { return c != 0; }, &S::comp)
                   .begin();
    assert(ret == it(a + 2));
    assert((*ret).comp == 0);
    assert((*ret).other == 1);
  }

  {
    // count projection and predicate invocation count
    {
      int a[]              = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;

      using it   = IteratorT<int*>;
      using sent = SentinelT<it>;

      auto ret =
          std::ranges::find_last_if_not(
              it(a),
              sent(it(a + 4)),
              [&](int i) {
                ++predicate_count;
                return i != 2;
              },
              [&](int i) {
                ++projection_count;
                return i;
              })
              .begin();
      assert(ret == it(a + 1));
      assert(*ret == 2);

      if constexpr (std::bidirectional_iterator<it>) {
        assert(predicate_count == 3);
        assert(projection_count == 3);
      } else {
        assert(predicate_count == 4);
        assert(projection_count == 4);
      }
    }
  }
}

struct NonConstComparable {
  friend constexpr bool operator!=(const NonConstComparable&, const NonConstComparable&) { return true; }
  friend constexpr bool operator!=(NonConstComparable&, NonConstComparable&) { return true; }
  friend constexpr bool operator!=(const NonConstComparable&, NonConstComparable&) { return true; }
  friend constexpr bool operator!=(NonConstComparable&, const NonConstComparable&) { return false; }
};

// TODO: this should really use `std::const_iterator`
template <class T>
struct add_const_to_ptr {
  using type = T;
};
template <class T>
struct add_const_to_ptr<T*> {
  using type = const T*;
};
template <class T>
using add_const_to_ptr_t = typename add_const_to_ptr<T>::type;

constexpr bool test() {
  test_iterator_classes<std::type_identity_t, std::type_identity_t>();
  test_iterator_classes<add_const_to_ptr_t, std::type_identity_t>();
  test_iterator_classes<contiguous_iterator, std::type_identity_t>();
  test_iterator_classes<random_access_iterator, std::type_identity_t>();
  test_iterator_classes<bidirectional_iterator, std::type_identity_t>();
  test_iterator_classes<forward_iterator, std::type_identity_t>();
  test_iterator_classes<forward_iterator, sentinel_wrapper>();

  {
    // check that projections are used properly and that they are called with the iterator directly
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = std::ranges::find_last_if_not(
                     a, a + 4, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; })
                     .begin();
      assert(ret == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret =
          std::ranges::find_last_if_not(a, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; }).begin();
      assert(ret == a + 3);
    }
  }

  {
    // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
        std::ranges::find_last_if_not(std::array{1, 2}, [](int) { return false; });
  }

  {
    // check that a subrange is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<std::ranges::subrange<int*>> auto ret =
        std::ranges::find_last_if_not(std::views::all(a), [](int) { return false; });
    assert(ret.begin() == a + 3);
    assert(*ret.begin() == 4);
  }

  {
    // check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparable a[] = {NonConstComparable{}};

      auto ret = std::ranges::find_last_if_not(a, a + 1, [](auto&& e) { return e != NonConstComparable{}; }).begin();
      assert(ret == a);
    }
    {
      NonConstComparable a[] = {NonConstComparable{}};

      auto ret = std::ranges::find_last_if_not(a, [](auto&& e) { return e != NonConstComparable{}; }).begin();
      assert(ret == a);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
