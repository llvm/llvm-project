//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Proj1 = identity, class Proj2 = identity,
//          indirect_strict_weak_order<projected<I1, Proj1>, projected<I2, Proj2>> Comp =
//            ranges::less>
//   constexpr bool includes(I1 first1, S1 last1, I2 first2, S2 last2, Comp comp = {},
//                           Proj1 proj1 = {}, Proj2 proj2 = {});                                   // Since C++20
//
// template<input_range R1, input_range R2, class Proj1 = identity,
//          class Proj2 = identity,
//          indirect_strict_weak_order<projected<iterator_t<R1>, Proj1>,
//                                     projected<iterator_t<R2>, Proj2>> Comp = ranges::less>
//   constexpr bool includes(R1&& r1, R2&& r2, Comp comp = {},
//                           Proj1 proj1 = {}, Proj2 proj2 = {});                                   // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <
    class Iter1 = int*,
    class Sent1 = int*,
    class Iter2 = int*,
    class Sent2 = int*,
    class Comp  = std::ranges::less,
    class Proj1 = std::identity,
    class Proj2 = std::identity>
concept HasIncludesIter =
    requires(Iter1&& iter1, Sent1&& sent1, Iter2&& iter2, Sent2&& sent2, Comp&& comp, Proj1&& proj1, Proj2&& proj2) {
      std::ranges::includes(
          std::forward<Iter1>(iter1),
          std::forward<Sent1>(sent1),
          std::forward<Iter2>(iter2),
          std::forward<Sent2>(sent2),
          std::forward<Comp>(comp),
          std::forward<Proj1>(proj1),
          std::forward<Proj2>(proj2));
    };

static_assert(HasIncludesIter<int*, int*, int*, int*>);

// !std::input_iterator<I1>
static_assert(!HasIncludesIter<InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S1, I1>
static_assert(!HasIncludesIter<int*, SentinelForNotSemiregular>);

// !std::input_iterator<I2>
static_assert(!HasIncludesIter<int*, int*, InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S2, I2>
static_assert(!HasIncludesIter<int*, int*, int*, SentinelForNotSemiregular>);

// !indirect_strict_weak_order<Comp, projected<I1, Proj1>, projected<I2, Proj2>>
struct NotAComparator {};
static_assert(!HasIncludesIter<int*, int*, int*, int*, NotAComparator>);

template <
    class Range1,
    class Range2,
    class Comp  = std::ranges::less,
    class Proj1 = std::identity,
    class Proj2 = std::identity>
concept HasIncludesRange =
    requires(Range1&& range1, Range2&& range2, Comp&& comp, Proj1&& proj1, Proj2&& proj2) {
      std::ranges::includes(
          std::forward<Range1>(range1),
          std::forward<Range2>(range2),
          std::forward<Comp>(comp),
          std::forward<Proj1>(proj1),
          std::forward<Proj2>(proj2));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasIncludesRange<R<int*>, R<int*>>);

// !std::input_range<R2>
static_assert(!HasIncludesRange<R<InputIteratorNotDerivedFrom>, R<int*>>);

// !std::input_range<R2>
static_assert(!HasIncludesRange<R<int*>, R<InputIteratorNotDerivedFrom>>);

// !indirect_strict_weak_order<Comp, projected<iterator_t<R1>, Proj1>,
//                                   projected<iterator_t<R2>, Proj2>>
static_assert(!HasIncludesRange<R<int*>, R<int*>, NotAComparator>);

template <class In1, class In2, template <class> class SentWrapper, std::size_t N1, std::size_t N2>
constexpr void testIncludesImpl(std::array<int, N1> in1, std::array<int, N2> in2, bool expected) {
  using Sent1 = SentWrapper<In1>;
  using Sent2 = SentWrapper<In2>;

  // iterator overload
  {
    std::same_as<bool> decltype(auto) result = std::ranges::includes(
        In1{in1.data()}, Sent1{In1{in1.data() + in1.size()}}, In2{in2.data()}, Sent2{In2{in2.data() + in2.size()}});
    assert(result == expected);
  }

  // range overload
  {
    std::ranges::subrange r1{In1{in1.data()}, Sent1{In1{in1.data() + in1.size()}}};
    std::ranges::subrange r2{In2{in2.data()}, Sent2{In2{in2.data() + in2.size()}}};
    std::same_as<bool> decltype(auto) result = std::ranges::includes(r1, r2);
    assert(result == expected);
  }
}

template <class In1, class In2, template <class> class SentWrapper>
constexpr void testImpl() {
  // range 1 shorter than range2
  {
    std::array in1{0, 1, 5, 6, 9, 10};
    std::array in2{3, 6, 7, 9, 13, 15, 100};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }
  // range 2 shorter than range 1 but not subsequence
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 1 and range 2 has the same length but different elements
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8, 15, 17, 19};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 1 == range 2
  {
    std::array in1{0, 1, 2};
    std::array in2{0, 1, 2};
    bool expected = true;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 2 is subsequence of range 1
  {
    std::array in1{8, 9, 10, 12, 13};
    std::array in2{8, 10};
    bool expected = true;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 1 is subsequence of range 2
  {
    std::array in1{0, 1, 1};
    std::array in2{0, 1, 1, 2, 5};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 2 is subsequence of range 1 with duplicated elements
  {
    std::array in1{8, 9, 10, 12, 12, 12};
    std::array in2{8, 12, 12};
    bool expected = true;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 2 is not a subsequence of range 1 because of duplicated elements
  {
    std::array in1{8, 9, 10, 12, 13};
    std::array in2{8, 10, 10};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 1 is empty
  {
    std::array<int, 0> in1{};
    std::array in2{3, 4, 5};
    bool expected = false;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // range 2 is empty
  {
    std::array in1{3, 4, 5};
    std::array<int, 0> in2{};
    bool expected = true;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }

  // both ranges are empty
  {
    std::array<int, 0> in1{};
    std::array<int, 0> in2{};
    bool expected = true;
    testIncludesImpl<In1, In2, SentWrapper>(in1, in2, expected);
  }
}

template <class Iter2, template <class> class SentWrapper>
constexpr void withAllPermutationsOfIter1() {
  // C++17 InputIterator may or may not satisfy std::input_iterator
  testImpl<cpp20_input_iterator<int*>, Iter2, sentinel_wrapper>();
  testImpl<forward_iterator<int*>, Iter2, SentWrapper>();
  testImpl<bidirectional_iterator<int*>, Iter2, SentWrapper>();
  testImpl<random_access_iterator<int*>, Iter2, SentWrapper>();
  testImpl<contiguous_iterator<int*>, Iter2, SentWrapper>();
  testImpl<int*, Iter2, SentWrapper>();
}

template <template <class> class SentWrapper>
constexpr void withAllPermutationsOfIter1AndIter2() {
  withAllPermutationsOfIter1<cpp20_input_iterator<int*>, sentinel_wrapper>();
  withAllPermutationsOfIter1<forward_iterator<int*>, SentWrapper>();
  withAllPermutationsOfIter1<bidirectional_iterator<int*>, SentWrapper>();
  withAllPermutationsOfIter1<random_access_iterator<int*>, SentWrapper>();
  withAllPermutationsOfIter1<contiguous_iterator<int*>, SentWrapper>();
  withAllPermutationsOfIter1<int*, SentWrapper>();
}

constexpr bool test() {
  withAllPermutationsOfIter1AndIter2<std::type_identity_t>();
  withAllPermutationsOfIter1AndIter2<sentinel_wrapper>();

  struct Data {
    int data;
  };

  // Test custom comparator
  {
    std::array r1{Data{4}, Data{8}, Data{12}};
    std::array r2{Data{4}, Data{12}};

    const auto comp = [](const Data& x, const Data& y) { return x.data < y.data; };
    bool expected   = true;

    // iterator overload
    {
      auto result = std::ranges::includes(r1.begin(), r1.end(), r2.begin(), r2.end(), comp);
      assert(result == expected);
    }

    // range overload
    {
      auto result = std::ranges::includes(r1, r2, comp);
      assert(result == expected);
    }
  }

  // Test custom projection
  {
    std::array r1{Data{4}, Data{8}, Data{12}};
    std::array r2{Data{4}, Data{9}};

    const auto proj = &Data::data;
    bool expected   = false;

    // iterator overload
    {
      auto result = std::ranges::includes(r1.begin(), r1.end(), r2.begin(), r2.end(), {}, proj, proj);
      assert(result == expected);
    }

    // range overload
    {
      auto result = std::ranges::includes(r1, r2, {}, proj, proj);
      assert(result == expected);
    }
  }

  // Complexity: At most 2 * ((last1 - first1) + (last2 - first2)) - 1
  // comparisons and applications of each projection.
  {
    struct CompProjs {
      std::size_t numberOfComp  = 0;
      std::size_t numberOfProj1 = 0;
      std::size_t numberOfProj2 = 0;

      constexpr auto comp() {
        return [this](int x, int y) {
          ++numberOfComp;
          return x < y;
        };
      }

      constexpr auto proj1() {
        return [this](const Data& d) {
          ++numberOfProj1;
          return d.data;
        };
      }

      constexpr auto proj2() {
        return [this](const Data& d) {
          ++numberOfProj2;
          return d.data;
        };
      }
    };
    std::array<Data, 4> r1{{{0}, {1}, {2}, {3}}};
    std::array<Data, 2> r2{{{4}, {5}}};
    const std::size_t maxOperation = 2 * (r1.size() + r2.size()) - 1;

    // iterator overload
    {
      CompProjs compProjs{};

      auto result = std::ranges::includes(
          r1.begin(), r1.end(), r2.begin(), r2.end(), compProjs.comp(), compProjs.proj1(), compProjs.proj2());
      assert(!result);
      assert(compProjs.numberOfComp < maxOperation);
      assert(compProjs.numberOfProj1 < maxOperation);
      assert(compProjs.numberOfProj2 < maxOperation);
    }

    // range overload
    {
      CompProjs compProjs{};

      auto result = std::ranges::includes(r1, r2, compProjs.comp(), compProjs.proj1(), compProjs.proj2());
      assert(!result);
      assert(compProjs.numberOfComp < maxOperation);
      assert(compProjs.numberOfProj1 < maxOperation);
      assert(compProjs.numberOfProj2 < maxOperation);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
