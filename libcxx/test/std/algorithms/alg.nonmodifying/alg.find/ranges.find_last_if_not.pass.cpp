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
//  constexpr subrange<I> ranges::find_last_if_not(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//  constexpr borrowed_subrange_t<R>
// ranges::find_last_if_not(R&& r, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct Predicate {
  bool operator()(int);
};

template <class It, class Sent = It>
concept HasFindLastIfNotIt = requires(It it, Sent sent) {
  std::ranges::find_last_if_not(it, sent, Predicate{});
};
static_assert(HasFindLastIfNotIt<int*>);
static_assert(!HasFindLastIfNotIt<InputIteratorNotDerivedFrom>);
static_assert(!HasFindLastIfNotIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasFindLastIfNotIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasFindLastIfNotIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindLastIfNotIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindLastIfNotIt<int*, int>);
static_assert(!HasFindLastIfNotIt<int, int*>);

template <class Pred>
concept HasFindLastIfNotPred = requires(int* it, Pred pred) {
  std::ranges::find_last_if_not(it, it, pred);
};

static_assert(HasFindLastIfNotPred<IndirectUnaryPredicate>);
static_assert(!HasFindLastIfNotPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindLastIfNotPred<IndirectUnaryPredicateNotPredicate>);

template <class R>
concept HasFindLastIfNotR = requires(R r) {
  std::ranges::find_last_if_not(r, Predicate{});
};
static_assert(HasFindLastIfNotR<std::array<int, 0>>);
static_assert(!HasFindLastIfNotR<int>);
static_assert(!HasFindLastIfNotR<InputRangeNotDerivedFrom>);
static_assert(!HasFindLastIfNotR<InputRangeNotIndirectlyReadable>);
static_assert(!HasFindLastIfNotR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasFindLastIfNotR<InputRangeNotSentinelSemiregular>);
static_assert(!HasFindLastIfNotR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
constexpr void test_iterators() {

  {// Test with an empty range

    {
      int a[] = {};
      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if_not(It(a), Sent(It(a)), [](int) { return false; });
      assert(ret.empty());
    }

    {
      int a[] = {};
      auto range = std::ranges::subrange(It(a), Sent(It(a)));
      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if_not(range, [](int) { return false; });
      assert(ret.empty());
    }

  }


  {// Test with a single element range

    {
      int a[] = {1};
      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if_not(It(a), Sent(It(a + 1)), [](int) { return false; });
      assert(base(ret.begin()) == a);
      assert(*ret.begin() == 1);
    }

    {
      int a[] = {1};
      std::same_as<std::ranges::borrowed_subrange_t<int (&)[1]>> auto ret = std::ranges::find_last_if_not(a, [](int) { return false; });
      assert(base(ret.begin()) == a);
      assert(*ret.begin() == 1);
    }

  }

  {// Test with a range where each element satisfies the predicate

    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if_not(It(a), Sent(It(a + 5)), [](int) { return false; });
      assert(base(ret.begin()) == a + 4);
      assert(*ret.begin() == 5);
    }

    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::borrowed_subrange_t<int (&)[5]>> auto ret = std::ranges::find_last_if_not(a, [](int) { return false; });
      assert(base(ret.begin()) == a + 4);
      assert(*ret.begin() == 5);
    }

  }

  {// Test with a range where no element satisfies the predicate

    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if_not(It(a), Sent(It(a + 5)), [](int) { return true; });
      //assert(base(ret.begin()) == a + 5);
      assert(ret.empty());
    }

    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::borrowed_subrange_t<int (&)[5]>> auto ret = std::ranges::find_last_if_not(a, [](int) { return true; });
      //assert(base(ret.begin()) == a + 5);
      assert(ret.empty());
    }

  }

}

struct NonConstComparableValue {
  friend constexpr bool operator==(const NonConstComparableValue&, const NonConstComparableValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableValue&, NonConstComparableValue&) { return false; }
  friend constexpr bool operator==(const NonConstComparableValue&, NonConstComparableValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableValue&, const NonConstComparableValue&) { return true; }
};

constexpr bool test() {
  test_iterators<const int*>();
  test_iterators<int*>();
  test_iterators<int*, const int*>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  {// check that projections are used properly and called with the reference to the element the iterator is pointing to

    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, a + 5, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; });
      assert(ret.data() == a + 3);
    }

    {
      int a[]  = {1, 2, 3, 4, 5};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; });
      assert(ret.data() == a + 3);
    }

  }

  {// check that the last element is returned

    {
      struct S{
        int comp;
        int other;
      };

      S a[] = {{0, 0}, {0, 2}, {0, 1}};
      std::same_as<std::ranges::borrowed_subrange_t<S (&)[3]>> auto ret = std::ranges::find_last_if_not(a, [](int i) { return i != 0; }, &S::comp);
      assert(ret.data() == a + 2);
      assert(ret.data()->comp == 0);
      assert(ret.data()->other == 1);
    }

    {
      struct S {
        int comp;
        int other;
      };

      S a[] = {{0, 0}, {0, 2}, {0, 1}};
      std::same_as<std::ranges::subrange<S *>> auto ret = std::ranges::find_last_if_not(a, a + 3, [](int i) { return i != 0; }, &S::comp);
      assert(ret.data() == a + 2);
      assert(ret.data()->comp == 0);
      assert(ret.data()->other == 1);
    }

  }

  {// check that end iterator is returned with no match

    {
      int a[] = {1, 1, 1};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, a + 3, [](int) { return false; });
      assert(ret.data() == a + 2);
    }

    {
      int a[] = {1, 1, 1};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, [](int) { return false; });
      assert(ret.data() == a + 2);
    }

  }

  {// check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
    std::ranges::find_last_if_not(std::array{1, 2}, [](int) { return true; });
  }

  {// check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(std::views::all(a), [](int) { return false; });
    assert(ret.data() == a + 3);
    assert(*ret.data() == 4);
  }

  {// check that std::invoke is used

    {
      struct S {
        int i;
      };

      S a[] = {S{1}, S{3}, S{2}};
      std::same_as<std::ranges::borrowed_subrange_t<S (&)[3]>> auto ret = std::ranges::find_last_if_not(a, [](int) { return true; }, &S::i);
      assert(ret.data() == a + 3);
    }

    {
      struct S {
        int i;
      };

      S a[] = {S{1}, S{3}, S{2}};
      std::same_as<std::ranges::subrange<S *>> auto ret = std::ranges::find_last_if_not(a, a + 3, [](int) { return true; }, &S::i);
      assert(ret.data() == a + 3);
    }

  }

  {// count projection and predicate invocation count

    {
      int a[] = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a,a + 4,
                                      [&](int i) { ++predicate_count; return i != 2; },
                                      [&](int i) { ++projection_count; return i; });
      assert(ret.data() == a + 1);
      assert(*ret.data() == 2);
      assert(predicate_count == 3);
      assert(projection_count == 3);
    }

    {
      int a[] = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a,
                                      [&](int i) { ++predicate_count; return i != 2; },
                                      [&](int i) { ++projection_count; return i; });
      assert(ret.data() == a + 1);
      assert(*ret.data() == 2);
      assert(predicate_count == 3);
      assert(projection_count == 3);
    }

  }

  {// check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparableValue a[] = {NonConstComparableValue{}};
      std::same_as<std::ranges::subrange<NonConstComparableValue *>> auto ret = std::ranges::find_last_if_not(a, a + 1, [](auto&& e) { return e != NonConstComparableValue{}; });
      assert(ret.data() == a);
    }

    {
      NonConstComparableValue a[] = {NonConstComparableValue{}};
      std::same_as<std::ranges::borrowed_subrange_t<NonConstComparableValue (&)[1]>> auto ret = std::ranges::find_last_if_not(a, [](auto&& e) { return e != NonConstComparableValue{}; });
      assert(ret.data() == a);
    }

  }

  {// check that the implicit conversion to bool works

    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, a + 4, [](const int& i) { return BooleanTestable{i != 3}; });
       assert(ret.data() == a + 2);
    }
    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ranges::subrange<int *>> auto ret = std::ranges::find_last_if_not(a, [](const int& b) { return BooleanTestable{b != 3}; });
      assert(ret.data() == a + 2);
    }

  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
