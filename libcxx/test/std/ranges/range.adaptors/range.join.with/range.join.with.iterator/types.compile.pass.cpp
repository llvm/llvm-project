//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// using iterator_concept = see below;
// using iterator_category = see below; // not always present
// using value_type = see below;
// using difference_type = see below;

#include <ranges>

#include <iterator>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

namespace test_iterator_concept {
template <template <class> class InnerIt>
using InnerRange = BasicView<std::vector<int>, ViewProperties{}, InnerIt>;

template <template <class> class It, template <class> class InnerIt>
using View = BasicView<std::vector<InnerRange<InnerIt>>, ViewProperties{}, It>;

template <template <class> class It, template <class> class InnerIt>
using RvalueView = BasicView<RvalueVector<InnerRange<InnerIt>>, ViewProperties{}, It>;

template <template <class> class It>
using Pattern = BasicView<std::vector<int>, ViewProperties{}, It>;

template <class V, class Pat>
using IteratorConcept = std::ranges::iterator_t<std::ranges::join_with_view<V, Pat>>::iterator_concept;

template <class V, class Pat, class Concept>
concept IteratorConceptIs = std::same_as<IteratorConcept<V, Pat>, Concept>;

// When `iterator<false>::iterator_concept` is `bidirectional_iterator_tag`
static_assert(IteratorConceptIs<View<bidirectional_iterator, bidirectional_iterator>,
                                Pattern<bidirectional_iterator>,
                                std::bidirectional_iterator_tag>);

// When `iterator<false>::iterator_concept` is `forward_iterator_tag`
static_assert(IteratorConceptIs<View<forward_iterator, bidirectional_iterator>,
                                Pattern<bidirectional_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<bidirectional_iterator, forward_iterator>,
                                Pattern<bidirectional_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<bidirectional_iterator, bidirectional_iterator>,
                                Pattern<forward_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<forward_iterator, forward_iterator>,
                                Pattern<bidirectional_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<forward_iterator, bidirectional_iterator>,
                                Pattern<forward_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<bidirectional_iterator, forward_iterator>,
                                Pattern<forward_iterator>,
                                std::forward_iterator_tag>);
static_assert(IteratorConceptIs<View<forward_iterator, forward_iterator>, //
                                Pattern<forward_iterator>,
                                std::forward_iterator_tag>);

// When `iterator<false>::iterator_concept` is `input_iterator_tag`
static_assert(IteratorConceptIs<View<DefaultCtorInputIter, forward_iterator>,
                                Pattern<forward_iterator>,
                                std::input_iterator_tag>);
static_assert(IteratorConceptIs<View<forward_iterator, DefaultCtorInputIter>,
                                Pattern<forward_iterator>,
                                std::input_iterator_tag>);
static_assert(IteratorConceptIs<View<DefaultCtorInputIter, DefaultCtorInputIter>,
                                Pattern<forward_iterator>,
                                std::input_iterator_tag>);
static_assert(IteratorConceptIs<RvalueView<bidirectional_iterator, bidirectional_iterator>,
                                Pattern<bidirectional_iterator>,
                                std::input_iterator_tag>);
static_assert(IteratorConceptIs<RvalueView<forward_iterator, forward_iterator>,
                                Pattern<forward_iterator>,
                                std::input_iterator_tag>);

template <class V, class Pat>
using ConstIteratorConcept = std::ranges::iterator_t<const std::ranges::join_with_view<V, Pat>>::iterator_concept;

template <class V, class Pat, class Concept>
concept ConstIteratorConceptIs = std::same_as<ConstIteratorConcept<V, Pat>, Concept>;

// When `iterator<true>::iterator_concept` is `bidirectional_iterator_tag`
static_assert(ConstIteratorConceptIs<View<bidirectional_iterator, bidirectional_iterator>,
                                     Pattern<bidirectional_iterator>,
                                     std::bidirectional_iterator_tag>);

// When `iterator<true>::iterator_concept` is `forward_iterator_tag`
static_assert(ConstIteratorConceptIs<View<forward_iterator, bidirectional_iterator>,
                                     Pattern<bidirectional_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<bidirectional_iterator, forward_iterator>,
                                     Pattern<bidirectional_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<bidirectional_iterator, bidirectional_iterator>,
                                     Pattern<forward_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<forward_iterator, forward_iterator>,
                                     Pattern<bidirectional_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<forward_iterator, bidirectional_iterator>,
                                     Pattern<forward_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<bidirectional_iterator, forward_iterator>,
                                     Pattern<forward_iterator>,
                                     std::forward_iterator_tag>);
static_assert(ConstIteratorConceptIs<View<forward_iterator, forward_iterator>,
                                     Pattern<forward_iterator>,
                                     std::forward_iterator_tag>);

// `iterator<true>::iterator_concept` cannot be `input_iterator_tag`
} // namespace test_iterator_concept

namespace test_iterator_category {
template <template <class> class InnerIt>
using InnerRange = BasicView<std::vector<float>, ViewProperties{}, InnerIt>;

template <bool Common, template <class> class InnerIt>
using MaybeCommonInnerRange = BasicView<std::vector<float>, ViewProperties{.common = Common}, InnerIt>;

template <template <class> class It, template <class> class InnerIt>
using View = BasicView<std::vector<InnerRange<InnerIt>>, ViewProperties{}, It>;

template <template <class> class It, template <class> class InnerIt>
using RvalueView = BasicView<RvalueVector<InnerRange<InnerIt>>, ViewProperties{}, It>;

template <bool Common, template <class> class It, bool CommonInner, template <class> class InnerIt>
using MaybeCommonView =
    BasicView<std::vector<MaybeCommonInnerRange<CommonInner, InnerIt>>, ViewProperties{.common = Common}, It>;

template <template <class> class It>
using Pattern = BasicView<std::vector<float>, ViewProperties{}, It>;

template <template <class> class It>
using RvaluePattern = BasicView<RvalueVector<float>, ViewProperties{}, It>;

template <bool Common, template <class> class It>
using MaybeCommonPattern = BasicView<std::vector<float>, ViewProperties{.common = Common}, It>;

template <class V, class Pattern>
using IteratorCategory = std::ranges::iterator_t<std::ranges::join_with_view<V, Pattern>>::iterator_category;

template <class V, class Pattern>
concept HasIteratorCategory = requires { typename IteratorCategory<V, Pattern>; };

template <class V, class Pat, class Category>
concept IteratorCategoryIs = std::same_as<IteratorCategory<V, Pat>, Category>;

// When `iterator<false>::iterator_category` is not defined
static_assert(!HasIteratorCategory<View<cpp20_input_iterator, forward_iterator>, Pattern<forward_iterator>>);
static_assert(!HasIteratorCategory<View<forward_iterator, cpp20_input_iterator>, Pattern<forward_iterator>>);
static_assert(!HasIteratorCategory<View<forward_iterator, forward_iterator>, Pattern<cpp20_input_iterator>>);
static_assert(!HasIteratorCategory<RvalueView<forward_iterator, forward_iterator>, Pattern<forward_iterator>>);
static_assert(HasIteratorCategory<View<forward_iterator, forward_iterator>, Pattern<forward_iterator>>);

// When
//   is_reference_v<common_reference_t<iter_reference_t<InnerIter>,
//                                     iter_reference_t<PatternIter>>>
// has different values for `iterator<false>`
static_assert(IteratorCategoryIs<View<forward_iterator, forward_iterator>,
                                 RvaluePattern<forward_iterator>,
                                 std::input_iterator_tag>);

// When `iterator<false>::iterator_category` is `bidirectional_iterator_tag`
static_assert(IteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                 MaybeCommonPattern<true, bidirectional_iterator>,
                                 std::bidirectional_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<false, bidirectional_iterator, true, bidirectional_iterator>,
                                 MaybeCommonPattern<true, bidirectional_iterator>,
                                 std::bidirectional_iterator_tag>);

// When `iterator<false>::iterator_category` is `forward_iterator_tag`
static_assert(IteratorCategoryIs<MaybeCommonView<true, forward_iterator, true, bidirectional_iterator>,
                                 MaybeCommonPattern<true, bidirectional_iterator>,
                                 std::forward_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, forward_iterator>,
                                 MaybeCommonPattern<true, bidirectional_iterator>,
                                 std::forward_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                 MaybeCommonPattern<true, forward_iterator>,
                                 std::forward_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, false, bidirectional_iterator>,
                                 MaybeCommonPattern<true, bidirectional_iterator>,
                                 std::forward_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                 MaybeCommonPattern<false, bidirectional_iterator>,
                                 std::forward_iterator_tag>);
static_assert(IteratorCategoryIs<MaybeCommonView<false, forward_iterator, false, forward_iterator>,
                                 MaybeCommonPattern<false, forward_iterator>,
                                 std::forward_iterator_tag>);

// When `iterator<false>::iterator_category` is `input_iterator_tag`
static_assert(IteratorCategoryIs<View<ForwardIteratorWithInputCategory, forward_iterator>,
                                 Pattern<forward_iterator>,
                                 std::input_iterator_tag>);
static_assert(IteratorCategoryIs<View<forward_iterator, ForwardIteratorWithInputCategory>,
                                 Pattern<forward_iterator>,
                                 std::input_iterator_tag>);
static_assert(IteratorCategoryIs<View<forward_iterator, forward_iterator>,
                                 Pattern<ForwardIteratorWithInputCategory>,
                                 std::input_iterator_tag>);
static_assert(IteratorCategoryIs<View<ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory>,
                                 Pattern<ForwardIteratorWithInputCategory>,
                                 std::input_iterator_tag>);

template <class V, class Pattern>
using ConstIteratorCategory = std::ranges::iterator_t<const std::ranges::join_with_view<V, Pattern>>::iterator_category;

template <class V, class Pattern>
concept HasConstIteratorCategory = requires { typename ConstIteratorCategory<V, Pattern>; };

template <class V, class Pat, class Category>
concept ConstIteratorCategoryIs = std::same_as<ConstIteratorCategory<V, Pat>, Category>;

// `iterator<true>::iterator_category` is not defined in those
// cases because `join_with_view<V, Pattern>` cannot const-accessed
static_assert(!HasConstIteratorCategory<View<cpp20_input_iterator, forward_iterator>, Pattern<forward_iterator>>);
static_assert(!HasConstIteratorCategory<View<forward_iterator, cpp20_input_iterator>, Pattern<forward_iterator>>);
static_assert(!HasConstIteratorCategory<View<forward_iterator, forward_iterator>, Pattern<cpp20_input_iterator>>);
static_assert(!HasConstIteratorCategory<RvalueView<forward_iterator, forward_iterator>, Pattern<forward_iterator>>);
static_assert(HasConstIteratorCategory<View<forward_iterator, forward_iterator>, Pattern<forward_iterator>>);

// When
//   is_reference_v<common_reference_t<iter_reference_t<InnerIter>,
//                                     iter_reference_t<PatternIter>>>
// has different values for `iterator<true>`
static_assert(ConstIteratorCategoryIs<View<forward_iterator, forward_iterator>,
                                      RvaluePattern<forward_iterator>,
                                      std::input_iterator_tag>);

// When `iterator<true>::iterator_category` is `bidirectional_iterator_tag`
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                      MaybeCommonPattern<true, bidirectional_iterator>,
                                      std::bidirectional_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<false, bidirectional_iterator, true, bidirectional_iterator>,
                                      MaybeCommonPattern<true, bidirectional_iterator>,
                                      std::bidirectional_iterator_tag>);
static_assert(ConstIteratorCategoryIs<
              BasicVectorView<
                  BasicVectorView<float, ViewProperties{.common = true}, forward_iterator, bidirectional_iterator>,
                  ViewProperties{.common = true},
                  forward_iterator,
                  bidirectional_iterator>,
              BasicVectorView<float, ViewProperties{.common = true}, forward_iterator, bidirectional_iterator>,
              std::bidirectional_iterator_tag>);

// When `iterator<true>::iterator_category` is `forward_iterator_tag`
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, forward_iterator, true, bidirectional_iterator>,
                                      MaybeCommonPattern<true, bidirectional_iterator>,
                                      std::forward_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, forward_iterator>,
                                      MaybeCommonPattern<true, bidirectional_iterator>,
                                      std::forward_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                      MaybeCommonPattern<true, forward_iterator>,
                                      std::forward_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, false, bidirectional_iterator>,
                                      MaybeCommonPattern<true, bidirectional_iterator>,
                                      std::forward_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<true, bidirectional_iterator, true, bidirectional_iterator>,
                                      MaybeCommonPattern<false, bidirectional_iterator>,
                                      std::forward_iterator_tag>);
static_assert(ConstIteratorCategoryIs<MaybeCommonView<false, forward_iterator, false, forward_iterator>,
                                      MaybeCommonPattern<false, forward_iterator>,
                                      std::forward_iterator_tag>);
static_assert(
    ConstIteratorCategoryIs<
        BasicVectorView<BasicVectorView<float, ViewProperties{}, ForwardIteratorWithInputCategory, forward_iterator>,
                        ViewProperties{},
                        ForwardIteratorWithInputCategory,
                        forward_iterator>,
        BasicVectorView<float, ViewProperties{}, ForwardIteratorWithInputCategory, forward_iterator>,
        std::forward_iterator_tag>);

// When `iterator<true>::iterator_category` is `input_iterator_tag`
static_assert(ConstIteratorCategoryIs<View<ForwardIteratorWithInputCategory, forward_iterator>,
                                      Pattern<forward_iterator>,
                                      std::input_iterator_tag>);
static_assert(ConstIteratorCategoryIs<View<forward_iterator, ForwardIteratorWithInputCategory>,
                                      Pattern<forward_iterator>,
                                      std::input_iterator_tag>);
static_assert(ConstIteratorCategoryIs<View<forward_iterator, forward_iterator>,
                                      Pattern<ForwardIteratorWithInputCategory>,
                                      std::input_iterator_tag>);
static_assert(ConstIteratorCategoryIs<View<ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory>,
                                      Pattern<ForwardIteratorWithInputCategory>,
                                      std::input_iterator_tag>);
static_assert(ConstIteratorCategoryIs<
              BasicVectorView<
                  BasicVectorView<float, ViewProperties{}, DefaultCtorInputIter, ForwardIteratorWithInputCategory>,
                  ViewProperties{},
                  DefaultCtorInputIter,
                  ForwardIteratorWithInputCategory>,
              BasicVectorView<float, ViewProperties{}, ForwardIteratorWithInputCategory>,
              std::input_iterator_tag>);
} // namespace test_iterator_category

namespace test_value_type {
template <class ValueType, class ConstValueType = ValueType>
struct View : std::ranges::view_base {
  struct InnerRange : std::ranges::view_base {
    ValueType* begin();
    ValueType* end();
    ConstValueType* begin() const;
    ConstValueType* end() const;
  };

  InnerRange* begin();
  InnerRange* end();
  const InnerRange* begin() const;
  const InnerRange* end() const;
};

template <class ValueType, class ConstValueType = ValueType>
using Pattern = View<ValueType, ConstValueType>::InnerRange;

template <class V, class Pat>
using IteratorValueType = std::ranges::iterator_t<std::ranges::join_with_view<V, Pat>>::value_type;

template <class V, class Pat, class ValueType>
concept IteratorValueTypeIs = std::same_as<IteratorValueType<V, Pat>, ValueType>;

// Test that `iterator<false>::value_type` is equal to
//   common_type_t<iter_value_t<InnerIter>, iter_value_t<PatternIter>>
static_assert(IteratorValueTypeIs<View<int>, Pattern<int>, int>);
static_assert(IteratorValueTypeIs<View<int>, Pattern<long>, long>);
static_assert(IteratorValueTypeIs<View<long>, Pattern<int>, long>);
static_assert(IteratorValueTypeIs<View<std::nullptr_t>, Pattern<void*>, void*>);
static_assert(IteratorValueTypeIs<View<std::tuple<long, int>>, Pattern<std::tuple<int, long>>, std::tuple<long, long>>);

template <class V, class Pat>
using ConstIteratorValueType = std::ranges::iterator_t<const std::ranges::join_with_view<V, Pat>>::value_type;

template <class V, class Pat, class ValueType>
concept ConstIteratorValueTypeIs = std::same_as<ConstIteratorValueType<V, Pat>, ValueType>;

// Test that `iterator<true>::value_type` is equal to
//   common_type_t<iter_value_t<InnerIter>, iter_value_t<PatternIter>>
static_assert(ConstIteratorValueTypeIs<View<int>, Pattern<int>, int>);
static_assert(ConstIteratorValueTypeIs<View<int>, Pattern<long>, long>);
static_assert(ConstIteratorValueTypeIs<View<long>, Pattern<int>, long>);
static_assert(ConstIteratorValueTypeIs<View<std::nullptr_t>, Pattern<void*>, void*>);
static_assert(
    ConstIteratorValueTypeIs<View<std::tuple<long, int>>, Pattern<std::tuple<int, long>>, std::tuple<long, long>>);

// Test value types of non-simple const ranges
static_assert(ConstIteratorValueTypeIs<View<short, int>, Pattern<short, int>, int>);
static_assert(ConstIteratorValueTypeIs<View<short, int>, Pattern<int, long>, long>);
static_assert(ConstIteratorValueTypeIs<View<int, long>, Pattern<short, int>, long>);
static_assert(ConstIteratorValueTypeIs<View<int, std::nullptr_t>, Pattern<int, void*>, void*>);
static_assert(ConstIteratorValueTypeIs<View<std::tuple<long, int>, std::pair<long, int>>,
                                       Pattern<std::tuple<int, long>, std::pair<int, long>>,
                                       std::pair<long, long>>);
} // namespace test_value_type

namespace test_difference_type {
template <class DifferenceType, class ValueType>
struct Iter {
  using value_type      = std::remove_const_t<ValueType>;
  using difference_type = DifferenceType;

  ValueType& operator*() const;
  Iter& operator++();
  Iter operator++(int);
  bool operator==(const Iter&) const;
};

static_assert(std::forward_iterator<Iter<int, void*>>);

template <class DifferenceType,
          class InnerDifferenceType,
          class ConstDifferenceType      = DifferenceType,
          class InnerConstDifferenceType = InnerDifferenceType>
struct View : std::ranges::view_base {
  struct InnerRange : std::ranges::view_base {
    Iter<InnerDifferenceType, float> begin();
    Iter<InnerDifferenceType, float> end();
    Iter<InnerConstDifferenceType, double> begin() const;
    Iter<InnerConstDifferenceType, double> end() const;
  };

  Iter<DifferenceType, InnerRange> begin();
  Iter<DifferenceType, InnerRange> end();
  Iter<ConstDifferenceType, const InnerRange> begin() const;
  Iter<ConstDifferenceType, const InnerRange> end() const;
};

template <class DifferenceType, class ConstDifferenceType = DifferenceType>
struct Pattern : std::ranges::view_base {
  Iter<DifferenceType, float> begin();
  Iter<DifferenceType, float> end();
  Iter<ConstDifferenceType, double> begin() const;
  Iter<ConstDifferenceType, double> end() const;
};

template <class V, class Pat>
using IteratorDifferenceType = std::ranges::iterator_t<std::ranges::join_with_view<V, Pat>>::difference_type;

template <class V, class Pat, class DifferenceType>
concept IteratorDifferenceTypeIs = std::same_as<IteratorDifferenceType<V, Pat>, DifferenceType>;

// Test that `iterator<false>::difference_type` is equal to
//   common_type_t<
//       iter_difference_t<OuterIter>,
//       iter_difference_t<InnerIter>,
//       iter_difference_t<PatternIter>>
static_assert(IteratorDifferenceTypeIs<View<int, int>, Pattern<int>, int>);
static_assert(IteratorDifferenceTypeIs<View<signed char, signed char>, Pattern<signed char>, signed char>);
static_assert(IteratorDifferenceTypeIs<View<short, short>, Pattern<short>, short>);
static_assert(IteratorDifferenceTypeIs<View<signed char, short>, Pattern<short>, int>);
static_assert(IteratorDifferenceTypeIs<View<signed char, short>, Pattern<int>, int>);
static_assert(IteratorDifferenceTypeIs<View<long long, long>, Pattern<int>, long long>);
static_assert(IteratorDifferenceTypeIs<View<long, long long>, Pattern<int>, long long>);

template <class V, class Pat>
using ConstIteratorDifferenceType = std::ranges::iterator_t<const std::ranges::join_with_view<V, Pat>>::difference_type;

template <class V, class Pat, class DifferenceType>
concept ConstIteratorDifferenceTypeIs = std::same_as<ConstIteratorDifferenceType<V, Pat>, DifferenceType>;

// Test that `iterator<true>::difference_type` is equal to
//   common_type_t<
//       iter_difference_t<OuterIter>,
//       iter_difference_t<InnerIter>,
//       iter_difference_t<PatternIter>>
static_assert(ConstIteratorDifferenceTypeIs<View<int, int>, Pattern<int>, int>);
static_assert(ConstIteratorDifferenceTypeIs<View<signed char, signed char>, Pattern<signed char>, signed char>);
static_assert(ConstIteratorDifferenceTypeIs<View<short, short>, Pattern<short>, short>);
static_assert(ConstIteratorDifferenceTypeIs<View<signed char, short>, Pattern<short>, int>);
static_assert(ConstIteratorDifferenceTypeIs<View<signed char, short>, Pattern<int>, int>);
static_assert(ConstIteratorDifferenceTypeIs<View<long long, long>, Pattern<int>, long long>);
static_assert(ConstIteratorDifferenceTypeIs<View<long, long long>, Pattern<int>, long long>);

// Test difference types of non-simple const ranges
static_assert(ConstIteratorDifferenceTypeIs<View<short, short, int, int>, Pattern<short, int>, int>);
static_assert(
    ConstIteratorDifferenceTypeIs<View<int, short, signed char, signed char>, Pattern<long, signed char>, signed char>);
static_assert(ConstIteratorDifferenceTypeIs<View<long, long long, signed char, short>, Pattern<long, short>, int>);
static_assert(ConstIteratorDifferenceTypeIs<View<short, short, long long, long>, Pattern<short, int>, long long>);
static_assert(ConstIteratorDifferenceTypeIs<View<signed char, signed char, long, long long>,
                                            Pattern<signed char, int>,
                                            long long>);
} // namespace test_difference_type
