//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>

// template<forward_iterator I1, sentinel_for<I1> S1, forward_iterator I2,
//        sentinel_for<I2> S2, class Proj1 = identity, class Proj2 = identity,
//        indirect_equivalence_relation<projected<I1, Proj1>,
//                                      projected<I2, Proj2>> Pred = ranges::equal_to>
// constexpr bool ranges::is_permutation(I1 first1, S1 last1, I2 first2, S2 last2,
//                                       Pred pred = {},
//                                       Proj1 proj1 = {}, Proj2 proj2 = {});                       // Since C++20
//
// template<forward_range R1, forward_range R2,
//        class Proj1 = identity, class Proj2 = identity,
//        indirect_equivalence_relation<projected<iterator_t<R1>, Proj1>,
//                                      projected<iterator_t<R2>, Proj2>> Pred = ranges::equal_to>
// constexpr bool ranges::is_permutation(R1&& r1, R2&& r2, Pred pred = {},
//                                       Proj1 proj1 = {}, Proj2 proj2 = {});                       // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <list>
#include <ranges>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = int*, class Iter2 = int*, class Sent2 = int*>
concept HasIsPermutationIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::is_permutation(first1, last1, first2, last2);
};

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasIsPermutationR = requires(Range1 range1, Range2 range2) {
  std::ranges::is_permutation(range1, range2);
};

static_assert(HasIsPermutationIt<int*>);
static_assert(!HasIsPermutationIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasIsPermutationIt<ForwardIteratorNotIncrementable>);
static_assert(!HasIsPermutationIt<int*, SentinelForNotSemiregular>);
static_assert(!HasIsPermutationIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasIsPermutationIt<int*, int*, ForwardIteratorNotDerivedFrom>);
static_assert(!HasIsPermutationIt<int*, int*, ForwardIteratorNotIncrementable>);
static_assert(!HasIsPermutationIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasIsPermutationIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);
// !indirect_equivalence_relation<Pred, projected<I1, Proj1>, projected<I2, Proj2>>;
static_assert(!HasIsPermutationIt<int*, int*, int**, int**>);

static_assert(HasIsPermutationR<UncheckedRange<int*>>);
static_assert(!HasIsPermutationR<ForwardRangeNotDerivedFrom>);
static_assert(!HasIsPermutationR<ForwardRangeNotIncrementable>);
static_assert(!HasIsPermutationR<int*, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasIsPermutationR<int*, ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasIsPermutationR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(!HasIsPermutationR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasIsPermutationR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasIsPermutationR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);
// !indirect_equivalence_relation<Pred, projected<iterator_t<I1>, Proj1>, projected<iterator_t<I2>, Proj2>>;
static_assert(!HasIsPermutationIt<UncheckedRange<int*>, UncheckedRange<int**>>);

template <int N, int M>
struct Data {
  std::array<int, N> input1;
  std::array<int, M> input2;
  bool expected;
};

template <class Iter1, class Sent1, class Iter2, class Sent2, int N, int M>
constexpr void test(Data<N, M> d) {
  {
    std::same_as<bool> decltype(auto) ret = std::ranges::is_permutation(Iter1(d.input1.data()),
                                                                        Sent1(Iter1(d.input1.data() + N)),
                                                                        Iter1(d.input2.data()),
                                                                        Sent1(Iter1(d.input2.data() + M)));
    assert(ret == d.expected);
  }
  {
    auto range1 = std::ranges::subrange(Iter1(d.input1.data()), Sent1(Iter1(d.input1.data() + N)));
    auto range2 = std::ranges::subrange(Iter1(d.input2.data()), Sent1(Iter1(d.input2.data() + M)));
    std::same_as<bool> decltype(auto) ret = std::ranges::is_permutation(range1, range2);
    assert(ret == d.expected);
  }
}

template <class Iter1, class Sent1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  // Ranges are identical.
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {1, 2, 3, 4}, .input2 = {1, 2, 3, 4}, .expected = true});

  // Ranges are reversed.
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {1, 2, 3, 4}, .input2 = {4, 3, 2, 1}, .expected = true});

  // Two elements are swapped.
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {4, 2, 3, 1}, .input2 = {1, 2, 3, 4}, .expected = true});

  // The first range is shorter.
  test<Iter1, Sent1, Iter2, Sent2, 4, 5>({.input1 = {4, 2, 3, 1}, .input2 = {4, 3, 2, 1, 5}, .expected = false});

  // The first range is longer.
  test<Iter1, Sent1, Iter2, Sent2, 5, 4>({.input1 = {4, 2, 3, 1, 5}, .input2 = {4, 3, 2, 1}, .expected = false});

  // The first range is empty.
  test<Iter1, Sent1, Iter2, Sent2, 0, 4>({.input1 = {}, .input2 = {4, 3, 2, 1}, .expected = false});

  // The second range is empty.
  test<Iter1, Sent1, Iter2, Sent2, 5, 0>({.input1 = {4, 2, 3, 1, 5}, .input2 = {}, .expected = false});

  // Both ranges are empty.
  test<Iter1, Sent1, Iter2, Sent2, 0, 0>({.input1 = {}, .input2 = {}, .expected = true});

  // 1-element range, same value.
  test<Iter1, Sent1, Iter2, Sent2, 1, 1>({.input1 = {1}, .input2 = {1}, .expected = true});

  // 1-element range, different values.
  test<Iter1, Sent1, Iter2, Sent2, 1, 1>({.input1 = {1}, .input2 = {2}, .expected = false});
}

template <class Iter1, class Sent1 = Iter1>
constexpr void test_iterators1() {
  test_iterators<Iter1, Sent1, forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<Iter1, Sent1, forward_iterator<int*>>();
  test_iterators<Iter1, Sent1, bidirectional_iterator<int*>>();
  test_iterators<Iter1, Sent1, random_access_iterator<int*>>();
  test_iterators<Iter1, Sent1, contiguous_iterator<int*>>();
  test_iterators<Iter1, Sent1, int*>();
  test_iterators<Iter1, Sent1, const int*>();
}

constexpr bool test() {
  test_iterators1<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators1<forward_iterator<int*>>();
  test_iterators1<bidirectional_iterator<int*>>();
  test_iterators1<random_access_iterator<int*>>();
  test_iterators1<contiguous_iterator<int*>>();
  test_iterators1<int*>();
  test_iterators1<const int*>();

  { // A custom comparator works.
    struct A {
      int a;
      constexpr bool pred(const A& rhs) const { return a == rhs.a; }
    };

    std::array in1 = {A{2}, A{3}, A{1}};
    std::array in2 = {A{1}, A{2}, A{3}};

    {
      auto ret = std::ranges::is_permutation(in1.begin(), in1.end(), in2.begin(), in2.end(), &A::pred);
      assert(ret);
    }

    {
      auto ret = std::ranges::is_permutation(in1, in2, &A::pred);
      assert(ret);
    }
  }

  { // A custom projection works.
    struct A {
      int a;

      constexpr bool operator==(const A&) const = default;

      constexpr A x2() const { return A{a * 2}; }
      constexpr A div2() const { return A{a / 2}; }
    };

    std::array in1 = {A{1}, A{2}, A{3}};  // [2, 4, 6] after applying `x2`.
    std::array in2 = {A{4}, A{8}, A{12}}; // [2, 4, 6] after applying `div2`.

    {
      auto ret = std::ranges::is_permutation(
          in1.begin(), in1.end(), in2.begin(), in2.end(), {}, &A::x2, &A::div2);
      assert(ret);
    }

    {
      auto ret = std::ranges::is_permutation(in1, in2, {}, &A::x2, &A::div2);
      assert(ret);
    }
  }


  { // Check that complexity requirements are met.
    int predCount = 0;
    int proj1Count = 0;
    int proj2Count = 0;
    auto reset_counters = [&] {
      predCount = proj1Count = proj2Count = 0;
    };

    counting_predicate pred(std::ranges::equal_to{}, predCount);
    counting_projection<> proj1(proj1Count);
    counting_projection<> proj2(proj2Count);

    {
      // 1. No applications of the corresponding predicate if `ForwardIterator1` and `ForwardIterator2` meet the
      //    requirements of random access iterators and `last1 - first1 != last2 - first2`.
      int a[] = {1, 2, 3, 4, 5};
      int b[] = {1, 2, 3, 4};
      // Make sure that the iterators have different types.
      auto b_begin = random_access_iterator<int*>(std::begin(b));
      auto b_end = random_access_iterator<int*>(std::end(b));

      {
        auto ret = std::ranges::is_permutation(a, a + 5, b_begin, b_end, pred, proj1, proj2);
        assert(!ret);

        assert(predCount == 0);
        assert(proj1Count == 0);
        assert(proj2Count == 0);
        reset_counters();
      }

      {
        auto ret = std::ranges::is_permutation(a, std::ranges::subrange(b_begin, b_end), pred, proj1, proj2);
        assert(!ret);

        assert(predCount == 0);
        assert(proj1Count == 0);
        assert(proj2Count == 0);
        reset_counters();
      }
    }

    // 2. Otherwise, exactly last1 - first1 applications of the corresponding predicate if
    // `equal(first1, last1, first2, last2, pred)` would return true.
    {
      int a[] = {1, 2, 3, 4, 5};
      int b[] = {1, 2, 3, 4, 5};
      int expected = 5;

      {
        auto ret = std::ranges::is_permutation(a, a + 5, b, b + 5, pred, proj1, proj2);
        assert(ret);

        assert(predCount == expected);
        assert(proj1Count == expected);
        assert(proj2Count == expected);
        reset_counters();
      }

      {
        auto ret = std::ranges::is_permutation(a, b, pred, proj1, proj2);
        assert(ret);

        assert(predCount == expected);
        assert(proj1Count == expected);
        assert(proj2Count == expected);
        reset_counters();
      }
    }

    // Note: we currently don't have the setup to test big-O complexity, but copying the requirement for completeness'
    // sake.
    // 3. Otherwise, at worst `O(N^2)`, where `N` has the value `last1 - first1`.
  }


  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
