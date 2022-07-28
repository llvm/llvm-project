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

// template<bidirectional_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//          class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   I inplace_merge(I first, I middle, S last, Comp comp = {}, Proj proj = {});                    // Since C++20
//
// template<bidirectional_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   borrowed_iterator_t<R>
//     inplace_merge(R&& r, iterator_t<R> middle, Comp comp = {},
//                   Proj proj = {});                                                               // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

template < class Iter,
           class Middle = Iter,
           class Sent   = sentinel_wrapper<std::remove_cvref_t<Iter>>,
           class Comp   = std::ranges::less,
           class Proj   = std::identity>
concept HasInplaceMergeIter =
    requires(Iter&& iter, Middle&& mid, Sent&& sent, Comp&& comp, Proj&& proj) {
      std::ranges::inplace_merge(
          std::forward<Iter>(iter),
          std::forward<Middle>(mid),
          std::forward<Sent>(sent),
          std::forward<Comp>(comp),
          std::forward<Proj>(proj));
    };

static_assert(HasInplaceMergeIter<int*, int*, int*>);

// !bidirectional_­iterator<I>
static_assert(!HasInplaceMergeIter<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasInplaceMergeIter<cpp20_input_iterator<int*>>);

// !sentinel_for<S, I>
static_assert(!HasInplaceMergeIter<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasInplaceMergeIter<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

// !sortable<I, Comp, Proj>
static_assert(!HasInplaceMergeIter<int*, int*, int*, ComparatorNotCopyable<int*>>);
static_assert(!HasInplaceMergeIter<const int*, const int*, const int*>);

template < class Range,
           class Middle = std::ranges::iterator_t<Range>,
           class Comp   = std::ranges::less,
           class Proj   = std::identity>
concept HasInplaceMergeRange =
    requires(Range&& r, Middle&& mid, Comp&& comp, Proj&& proj) {
      std::ranges::inplace_merge(
          std::forward<Range>(r), std::forward<Middle>(mid), std::forward<Comp>(comp), std::forward<Proj>(proj));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasInplaceMergeRange<R<int*>, int*>);

// !bidirectional_range<R>
static_assert(!HasInplaceMergeRange<R<cpp20_input_iterator<int*>>>);
static_assert(!HasInplaceMergeRange<R<BidirectionalIteratorNotDecrementable>>);

// !sortable<iterator_t<R>, Comp, Proj>
static_assert(!HasInplaceMergeRange<R<int*>, int*, ComparatorNotCopyable<int*>>);
static_assert(!HasInplaceMergeIter<R<const int*>, const int*>);

template <class In, template <class> class SentWrapper, std::size_t N1, std::size_t N2>
void testInplaceMergeImpl(std::array<int, N1> input, int midIdx, std::array<int, N2> expected) {
  std::is_sorted(input.begin(), input.begin() + midIdx);
  std::is_sorted(input.begin() + midIdx, input.end());
  std::is_sorted(expected.begin(), expected.end());

  using Sent = SentWrapper<In>;

  // iterator overload
  {
    auto in = input;
    std::same_as<In> decltype(auto) result =
        std::ranges::inplace_merge(In{in.data()}, In{in.data() + midIdx}, Sent{In{in.data() + in.size()}});
    assert(std::ranges::equal(in, expected));
    assert(base(result) == in.data() + in.size());
  }

  // range overload
  {
    auto in = input;
    std::ranges::subrange r{In{in.data()}, Sent{In{in.data() + in.size()}}};
    std::same_as<In> decltype(auto) result = std::ranges::inplace_merge(r, In{in.data() + midIdx});
    assert(std::ranges::equal(in, expected));
    assert(base(result) == in.data() + in.size());
  }
}

template <class In, template <class> class SentWrapper>
void testImpl() {
  // sorted range
  {
    std::array in{0, 1, 5, 6, 9, 10};
    std::array expected = in;
    testInplaceMergeImpl<In, SentWrapper>(in, 3, expected);
  }

  // [first, mid) is longer
  {
    std::array in{0, 5, 9, 15, 18, 22, 2, 4, 6, 10};
    std::array expected = {0, 2, 4, 5, 6, 9, 10, 15, 18, 22};
    testInplaceMergeImpl<In, SentWrapper>(in, 6, expected);
  }

  // [first, mid) is shorter
  {
    std::array in{0, 5, 9, 2, 4, 6, 10};
    std::array expected = {0, 2, 4, 5, 6, 9, 10};
    testInplaceMergeImpl<In, SentWrapper>(in, 3, expected);
  }

  // [first, mid) == [mid, last)
  {
    std::array in{0, 5, 9, 0, 5, 9};
    std::array expected = {0, 0, 5, 5, 9, 9};
    testInplaceMergeImpl<In, SentWrapper>(in, 3, expected);
  }

  // duplicates within each range
  {
    std::array in{1, 5, 5, 2, 9, 9, 9};
    std::array expected = {1, 2, 5, 5, 9, 9, 9};
    testInplaceMergeImpl<In, SentWrapper>(in, 3, expected);
  }

  // all the same
  {
    std::array in{5, 5, 5, 5, 5, 5, 5, 5};
    std::array expected = in;
    testInplaceMergeImpl<In, SentWrapper>(in, 5, expected);
  }

  // [first, mid) is empty (mid == begin)
  {
    std::array in{0, 1, 5, 6, 9, 10};
    std::array expected = in;
    testInplaceMergeImpl<In, SentWrapper>(in, 0, expected);
  }

  // [mid, last] is empty (mid == end)
  {
    std::array in{0, 1, 5, 6, 9, 10};
    std::array expected = in;
    testInplaceMergeImpl<In, SentWrapper>(in, 6, expected);
  }

  // both empty
  {
    std::array<int, 0> in{};
    std::array expected = in;
    testInplaceMergeImpl<In, SentWrapper>(in, 0, expected);
  }

  // mid == first + 1
  {
    std::array in{9, 2, 5, 7, 10};
    std::array expected{2, 5, 7, 9, 10};
    testInplaceMergeImpl<In, SentWrapper>(in, 1, expected);
  }

  // mid == last - 1
  {
    std::array in{2, 5, 7, 10, 9};
    std::array expected{2, 5, 7, 9, 10};
    testInplaceMergeImpl<In, SentWrapper>(in, 4, expected);
  }
}

template < template <class> class SentWrapper>
void withAllPermutationsOfIter() {
  testImpl<bidirectional_iterator<int*>, SentWrapper>();
  testImpl<random_access_iterator<int*>, SentWrapper>();
  testImpl<contiguous_iterator<int*>, SentWrapper>();
  testImpl<int*, SentWrapper>();
}

bool test() {
  withAllPermutationsOfIter<std::type_identity_t>();
  withAllPermutationsOfIter<sentinel_wrapper>();

  struct Data {
    int data;
  };

  const auto equal = [](const Data& x, const Data& y) { return x.data == y.data; };
  // Test custom comparator
  {
    std::array<Data, 4> input{{{4}, {8}, {2}, {5}}};
    std::array<Data, 4> expected{{{2}, {4}, {5}, {8}}};
    const auto comp = [](const Data& x, const Data& y) { return x.data < y.data; };

    // iterator overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in.begin(), in.begin() + 2, in.end(), comp);
      assert(std::ranges::equal(in, expected, equal));
      assert(result == in.end());
    }

    // range overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in, in.begin() + 2, comp);
      assert(std::ranges::equal(in, expected, equal));
      assert(result == in.end());
    }
  }

  // Test custom projection
  {
    std::array<Data, 4> input{{{4}, {8}, {2}, {5}}};
    std::array<Data, 4> expected{{{2}, {4}, {5}, {8}}};

    const auto proj = &Data::data;

    // iterator overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in.begin(), in.begin() + 2, in.end(), {}, proj);
      assert(std::ranges::equal(in, expected, equal));
      assert(result == in.end());
    }

    // range overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in, in.begin() + 2, {}, proj);
      assert(std::ranges::equal(in, expected, equal));
      assert(result == in.end());
    }
  }

  // Remarks: Stable.
  {
    struct IntAndID {
      int data;
      int id;
      constexpr auto operator<=>(const IntAndID& rhs) const { return data <=> rhs.data; }
      constexpr auto operator==(const IntAndID& rhs) const { return data == rhs.data; }
    };
    std::array<IntAndID, 6> input{{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}}};

    // iterator overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in.begin(), in.begin() + 3, in.end());
      assert(std::ranges::equal(in, std::array{0, 0, 1, 1, 2, 2}, {}, &IntAndID::data));
      assert(std::ranges::equal(in, std::array{0, 1, 0, 1, 0, 1}, {}, &IntAndID::id));
      assert(result == in.end());
    }

    // range overload
    {
      auto in     = input;
      auto result = std::ranges::inplace_merge(in, in.begin() + 3);
      assert(std::ranges::equal(in, std::array{0, 0, 1, 1, 2, 2}, {}, &IntAndID::data));
      assert(std::ranges::equal(in, std::array{0, 1, 0, 1, 0, 1}, {}, &IntAndID::id));
      assert(result == in.end());
    }
  }

  // Complexity: Let N = last - first :
  //   - For the overloads with no ExecutionPolicy, and if enough
  //     additional memory is available, exactly N − 1 comparisons.
  //   - Otherwise, O(NlogN) comparisons.
  // In either case, twice as many projections as comparisons.
  {
    std::array input{1, 2, 3, 3, 3, 7, 7, 2, 2, 5, 5, 6, 6};
    std::array expected{1, 2, 2, 2, 3, 3, 3, 5, 5, 6, 6, 7, 7};
    auto mid = 7;
    // iterator overload
    {
      auto in          = input;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::inplace_merge(
          in.begin(),
          in.begin() + mid,
          in.end(),
          counting_predicate{std::ranges::less{}, numberOfComp},
          counting_projection{numberOfProj});
      assert(std::ranges::equal(in, expected));
      assert(result == in.end());

      // the spec specifies exactly N-1 comparison but we actually
      // do not invoke as many times as specified
      assert(numberOfComp <= static_cast<int>(in.size() - 1));
      assert(numberOfProj <= 2 * numberOfComp);
    }
    // range overload
    {
      auto in          = input;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::inplace_merge(
          in,
          in.begin() + mid,
          counting_predicate{std::ranges::less{}, numberOfComp},
          counting_projection{numberOfProj});
      assert(std::ranges::equal(in, expected));
      assert(result == in.end());
      assert(numberOfComp <= static_cast<int>(in.size() - 1));
      assert(numberOfProj <= 2 * numberOfComp);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  // inplace_merge is not constexpr in the latest finished Standard (C++20)

  return 0;
}
