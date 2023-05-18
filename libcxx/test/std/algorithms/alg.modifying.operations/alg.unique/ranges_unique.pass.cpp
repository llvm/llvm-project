//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<permutable I, sentinel_for<I> S, class Proj = identity,
//          indirect_equivalence_relation<projected<I, Proj>> C = ranges::equal_to>
//   constexpr subrange<I> unique(I first, S last, C comp = {}, Proj proj = {});                    // Since C++20
//
// template<forward_range R, class Proj = identity,
//          indirect_equivalence_relation<projected<iterator_t<R>, Proj>> C = ranges::equal_to>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R>
//     unique(R&& r, C comp = {}, Proj proj = {});                                                  // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

template <class Iter = int*, class Sent = int*, class Comp = std::ranges::equal_to, class Proj = std::identity>
concept HasUniqueIter =
    requires(Iter&& iter, Sent&& sent, Comp&& comp, Proj&& proj) {
      std::ranges::unique(
          std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Comp>(comp), std::forward<Proj>(proj));
    };

static_assert(HasUniqueIter<int*, int*>);

// !permutable<I>
static_assert(!HasUniqueIter<PermutableNotForwardIterator>);
static_assert(!HasUniqueIter<PermutableNotSwappable>);

// !sentinel_for<S, I>
static_assert(!HasUniqueIter<int*, SentinelForNotSemiregular>);

// !indirect_equivalence_relation<Comp, projected<I, Proj>>
static_assert(!HasUniqueIter<int*, int*, ComparatorNotCopyable<int>>);

template <class Range, class Comp = std::ranges::equal_to, class Proj = std::identity>
concept HasUniqueRange =
    requires(Range&& range, Comp&& comp, Proj&& proj) {
      std::ranges::unique(std::forward<Range>(range), std::forward<Comp>(comp), std::forward<Proj>(proj));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasUniqueRange<R<int*>>);

// !forward_range<R>
static_assert(!HasUniqueRange<ForwardRangeNotDerivedFrom>);
static_assert(!HasUniqueRange<ForwardRangeNotIncrementable>);

// permutable<ranges::iterator_t<R>>
static_assert(!HasUniqueRange<R<PermutableNotForwardIterator>>);
static_assert(!HasUniqueRange<R<PermutableNotSwappable>>);

// !indirect_equivalence_relation<Comp, projected<ranges::iterator_t<R>, Proj>>
static_assert(!HasUniqueRange<R<int*>, ComparatorNotCopyable<int>>);

template <class Iter, template <class> class SentWrapper, std::size_t N1, std::size_t N2>
constexpr void testUniqueImpl(std::array<int, N1> input, std::array<int, N2> expected) {
  using Sent = SentWrapper<Iter>;

  // iterator overload
  {
    auto in = input;
    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
        std::ranges::unique(Iter{in.data()}, Sent{Iter{in.data() + in.size()}});
    assert(std::ranges::equal(std::ranges::subrange<Iter>{Iter{in.data()}, result.begin()}, expected));
    assert(base(result.end()) == in.data() + in.size());
  }

  // range overload
  {
    auto in = input;
    std::ranges::subrange r{Iter{in.data()}, Sent{Iter{in.data() + in.size()}}};
    std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::unique(r);
    assert(std::ranges::equal(std::ranges::subrange<Iter>{Iter{in.data()}, result.begin()}, expected));
    assert(base(result.end()) == in.data() + in.size());
  }
}

template <class Iter, template <class> class SentWrapper>
constexpr void testImpl() {
  // no consecutive elements
  {
    std::array in{1, 2, 3, 2, 1};
    std::array expected{1, 2, 3, 2, 1};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
  }

  // one group of consecutive elements
  {
    std::array in{2, 3, 3, 3, 4, 3};
    std::array expected{2, 3, 4, 3};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
  }

  // multiple groups of consecutive elements
  {
    std::array in{2, 3, 3, 3, 4, 3, 3, 5, 5, 5};
    std::array expected{2, 3, 4, 3, 5};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
  }

  // all the same
  {
    std::array in{1, 1, 1, 1, 1, 1};
    std::array expected{1};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
  }

  // empty range
  {
    std::array<int, 0> in{};
    std::array<int, 0> expected{};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
  }

  // single element range
    std::array in{1};
    std::array expected{1};
    testUniqueImpl<Iter, SentWrapper>(in, expected);
}

template <template <class> class SentWrapper>
constexpr void withAllPermutationsOfIter() {
  testImpl<forward_iterator<int*>, SentWrapper>();
  testImpl<bidirectional_iterator<int*>, SentWrapper>();
  testImpl<random_access_iterator<int*>, SentWrapper>();
  testImpl<contiguous_iterator<int*>, SentWrapper>();
  testImpl<int*, SentWrapper>();
}

constexpr bool test() {
  withAllPermutationsOfIter<std::type_identity_t>();
  withAllPermutationsOfIter<sentinel_wrapper>();

  struct Data {
    int data;
  };

  // Test custom comparator
  {
    std::array input{Data{4}, Data{8}, Data{8}, Data{8}};
    std::array expected{Data{4}, Data{8}};
    const auto comp = [](const Data& x, const Data& y) { return x.data == y.data; };

    // iterator overload
    {
      auto in     = input;
      auto result = std::ranges::unique(in.begin(), in.end(), comp);
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end(), comp));
      assert(base(result.end()) == in.end());
    }

    // range overload
    {
      auto in     = input;
      auto result = std::ranges::unique(in, comp);
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end(), comp));
      assert(base(result.end()) == in.end());
    }
  }

  // Test custom projection
  {
    std::array input{Data{4}, Data{8}, Data{8}, Data{8}};
    std::array expected{Data{4}, Data{8}};

    const auto proj = &Data::data;

    // iterator overload
    {
      auto in     = input;
      auto result = std::ranges::unique(in.begin(), in.end(), {}, proj);
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end(), {}, proj, proj));
      assert(base(result.end()) == in.end());
    }

    // range overload
    {
      auto in     = input;
      auto result = std::ranges::unique(in, {}, proj);
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end(), {}, proj, proj));
      assert(base(result.end()) == in.end());
    }
  }

  // Complexity: For nonempty ranges, exactly (last - first) - 1 applications of the corresponding predicate
  // and no more than twice as many applications of any projection.
  {
    std::array input{1, 2, 3, 3, 3, 4, 3, 3, 5, 5, 6, 6, 1};
    std::array expected{1, 2, 3, 4, 3, 5, 6, 1};
    // iterator overload
    {
      auto in          = input;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::unique(
          in.begin(),
          in.end(),
          counting_predicate{std::ranges::equal_to{}, numberOfComp},
          counting_projection{numberOfProj});
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end()));
      assert(base(result.end()) == in.end());
      assert(numberOfComp == in.size() - 1);
      assert(numberOfProj <= static_cast<int>(2 * (in.size() - 1)));
    }
    // range overload
    {
      auto in          = input;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::unique(
          in, counting_predicate{std::ranges::equal_to{}, numberOfComp}, counting_projection{numberOfProj});
      assert(std::ranges::equal(in.begin(), result.begin(), expected.begin(), expected.end()));
      assert(base(result.end()) == in.end());
      assert(numberOfComp == in.size() - 1);
      assert(numberOfProj <= static_cast<int>(2 * (in.size() - 1)));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
