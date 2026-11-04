//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_TYPES_H

#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class Tp>
void pass_value(Tp);

template <class Tp, class... Args>
concept ConstructionIsExplicit = std::constructible_from<Tp, Args...> && !requires(Args&&... args) {
  pass_value<Tp>({std::forward<Args>(args)...});
};

struct ViewProperties {
  bool simple = false;
  bool common = true;
};

template <std::ranges::input_range Data,
          ViewProperties Prop,
          template <class...> class It,
          template <class...> class ConstIt = It>
class BasicView : public std::ranges::view_base {
  Data data_;

public:
  constexpr BasicView()
    requires std::default_initializable<Data>
  = default;

  template <class R>
  constexpr explicit BasicView(R&& r)
    requires requires { std::ranges::to<Data>(std::forward<R>(r)); }
      /*******/ : data_(std::ranges::to<Data>(std::forward<R>(r))) {}

  constexpr explicit BasicView(std::initializer_list<std::ranges::range_value_t<Data>> il)
      : data_(std::ranges::to<Data>(il)) {}

  constexpr auto begin()
    requires(!Prop.simple)
  {
    return It(data_.begin());
  }

  constexpr auto end()
    requires(!Prop.simple)
  {
    if constexpr (Prop.common)
      return It(data_.end());
    else
      return sentinel_wrapper(It(data_.end()));
  }

  constexpr auto begin() const { return ConstIt(data_.begin()); }

  constexpr auto end() const {
    if constexpr (Prop.common)
      return ConstIt(data_.end());
    else
      return sentinel_wrapper(ConstIt(data_.end()));
  }
};

template <class Tp, ViewProperties Prop, template <class...> class It, template <class...> class ConstIt = It>
using BasicVectorView = BasicView<std::vector<Tp>, Prop, It, ConstIt>;

struct AsPrvalue {
  template <class Tp>
  constexpr auto operator()(Tp&& t) const {
    return std::forward<Tp>(t);
  }
};

template <class Tp>
class RvalueVector {
  using Vec = std::vector<Tp>;
  std::ranges::transform_view<std::ranges::owning_view<Vec>, AsPrvalue> range_;

public:
  constexpr RvalueVector() = default;
  constexpr explicit RvalueVector(Vec vec) : range_(std::move(vec), AsPrvalue{}) {}
  constexpr explicit RvalueVector(std::initializer_list<Tp> il) : RvalueVector(Vec(il)) {}

  constexpr auto begin() { return range_.begin(); }
  constexpr auto end() { return range_.end(); }
  constexpr auto begin() const { return range_.begin(); }
  constexpr auto end() const { return range_.end(); }
};

template <class It>
class DefaultCtorInputIter {
  It it_ = It();

public:
  using value_type      = std::iter_value_t<It>;
  using difference_type = std::iter_difference_t<It>;

  DefaultCtorInputIter() = default;
  constexpr explicit DefaultCtorInputIter(It it) : it_(it) {}

  constexpr DefaultCtorInputIter& operator++() {
    ++it_;
    return *this;
  }

  constexpr void operator++(int) { ++*this; }
  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr bool operator==(const DefaultCtorInputIter&) const = default;
};

template <class It>
DefaultCtorInputIter(It) -> DefaultCtorInputIter<It>;

template <class Tp>
class InputRangeButOutputWhenConst {
  using Vec = std::vector<Tp>;
  std::ranges::ref_view<Vec> range_;

public:
  constexpr explicit InputRangeButOutputWhenConst(Vec& vec) : range_(vec) {}

  constexpr auto begin() { return cpp20_input_iterator(range_.begin()); }
  constexpr auto end() { return sentinel_wrapper(cpp20_input_iterator(range_.end())); }
  constexpr auto begin() const { return cpp20_output_iterator(range_.begin()); }
  constexpr auto end() const { return sentinel_wrapper(cpp20_output_iterator(range_.end())); }
};

template <class Tp>
using ForwardViewButInputWhenConst =
    BasicVectorView<Tp, ViewProperties{.common = false}, forward_iterator, cpp20_input_iterator>;

template <class It>
class ForwardIteratorWithInputCategory {
  It it_ = It();

public:
  using value_type        = std::iter_value_t<It>;
  using difference_type   = std::iter_difference_t<It>;
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::input_iterator_tag;

  ForwardIteratorWithInputCategory() = default;
  explicit ForwardIteratorWithInputCategory(It it);

  std::iter_reference_t<It> operator*() const;
  ForwardIteratorWithInputCategory& operator++();
  ForwardIteratorWithInputCategory operator++(int);
  bool operator==(const ForwardIteratorWithInputCategory&) const;
};

template <class It>
explicit ForwardIteratorWithInputCategory(It) -> ForwardIteratorWithInputCategory<It>;

template <class It>
class EqComparableInputIter {
  It it_;

public:
  using value_type      = std::iter_value_t<It>;
  using difference_type = std::iter_difference_t<It>;

  constexpr explicit EqComparableInputIter(It it) : it_(it) {}

  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr EqComparableInputIter& operator++() {
    ++it_;
    return *this;
  }
  constexpr void operator++(int) { ++it_; }

  friend constexpr It base(const EqComparableInputIter& i) { return i.it_; }

  friend constexpr bool operator==(const EqComparableInputIter& left, const EqComparableInputIter& right) {
    return left.it_ == right.it_;
  }
};

template <class It>
EqComparableInputIter(It) -> EqComparableInputIter<It>;

template <class Val>
struct ConstOppositeView : std::ranges::view_base {
  const Val* begin();
  sentinel_wrapper<const Val*> end();
  Val* begin() const;
  sentinel_wrapper<Val*> end() const;
};

namespace lwg4074 { // Helpers for LWG-4074 ("compatible-joinable-ranges is underconstrained")
struct CommonReference;

struct Value {
  Value(int);
};

struct Reference {
  operator Value() const;
  operator CommonReference() const;
};

struct CommonReference {
  CommonReference(int);
};

struct Iter {
  using value_type      = Value;
  using difference_type = std::ptrdiff_t;

  Iter& operator++();
  Iter operator++(int);
  Reference operator*() const;
  bool operator==(const Iter&) const;
};

struct PatternWithProxyConstAccess {
  int* begin();
  int* end();
  Iter begin() const;
  Iter end() const;
};
} // namespace lwg4074

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<lwg4074::Reference, int, TQual, UQual> {
  using type = lwg4074::CommonReference;
};

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<int, lwg4074::Reference, TQual, UQual> {
  using type = lwg4074::CommonReference;
};

namespace selftest {
using BV1 = BasicView<std::string, ViewProperties{.simple = true}, forward_iterator>;
static_assert(std::ranges::forward_range<BV1>);
static_assert(!std::ranges::bidirectional_range<BV1>);
static_assert(std::ranges::common_range<BV1>);
static_assert(simple_view<BV1>);

using BV2 =
    BasicView<RvalueVector<std::string>, ViewProperties{.simple = false, .common = false}, cpp20_input_iterator>;
static_assert(std::ranges::input_range<BV2>);
static_assert(!std::ranges::forward_range<BV2>);
static_assert(!std::ranges::common_range<BV2>);
static_assert(!std::is_reference_v<std::ranges::range_reference_t<BV2>>);
static_assert(!simple_view<BV2>);

using RV = RvalueVector<int>;
static_assert(std::movable<RV>);
static_assert(std::ranges::random_access_range<RV>);
static_assert(std::ranges::random_access_range<const RV>);
static_assert(!std::is_reference_v<std::ranges::range_reference_t<RV>>);
static_assert(!std::is_reference_v<std::ranges::range_reference_t<const RV>>);

using DCII = DefaultCtorInputIter<int*>;
static_assert(std::default_initializable<DCII>);
static_assert(std::sentinel_for<DCII, DCII>);
static_assert(std::input_iterator<DCII>);
static_assert(!std::forward_iterator<DCII>);

using IRBOWC = InputRangeButOutputWhenConst<int>;
static_assert(std::ranges::input_range<IRBOWC>);
static_assert(std::ranges::output_range<const IRBOWC&, int>);

using FVBIWC = ForwardViewButInputWhenConst<int>;
static_assert(std::default_initializable<FVBIWC>);
static_assert(std::ranges::view<FVBIWC>);
static_assert(std::ranges::forward_range<FVBIWC>);
static_assert(!std::ranges::common_range<FVBIWC>);
static_assert(std::ranges::input_range<const FVBIWC&>);
static_assert(!std::ranges::forward_range<const FVBIWC&>);
static_assert(!std::ranges::common_range<const FVBIWC&>);

using FIWIC = ForwardIteratorWithInputCategory<long*>;
static_assert(std::forward_iterator<FIWIC>);
static_assert(std::same_as<FIWIC::iterator_category, std::input_iterator_tag>);
static_assert(std::same_as<FIWIC::iterator_category, std::iterator_traits<FIWIC>::iterator_category>);

using ECII = EqComparableInputIter<int*>;
static_assert(std::input_iterator<ECII>);
static_assert(!std::forward_iterator<ECII>);
static_assert(std::equality_comparable<ECII>);

using COV = ConstOppositeView<int>;
static_assert(std::ranges::view<COV>);
static_assert(std::ranges::range<const COV>);
static_assert(!std::ranges::common_range<COV>);
static_assert(!std::ranges::common_range<const COV>);
static_assert(std::convertible_to<std::ranges::iterator_t<const COV>, std::ranges::iterator_t<COV>>);
static_assert(!std::convertible_to<std::ranges::iterator_t<COV>, std::ranges::iterator_t<const COV>>);

static_assert(std::common_with<lwg4074::Value, int>);
static_assert(std::common_with<lwg4074::Value, lwg4074::Reference>);
static_assert(std::common_reference_with<lwg4074::Reference, int&>);
static_assert(std::common_reference_with<lwg4074::Reference, lwg4074::CommonReference>);
static_assert(std::forward_iterator<lwg4074::Iter>);
static_assert(std::ranges::forward_range<lwg4074::PatternWithProxyConstAccess>);
static_assert(std::ranges::forward_range<const lwg4074::PatternWithProxyConstAccess>);
} // namespace selftest

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_TYPES_H
