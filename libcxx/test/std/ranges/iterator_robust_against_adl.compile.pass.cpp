//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !c++experimental

// ADL call with nested iterators of views should not look up base's view's
// namespace

#include <ranges>
#include <tuple>

#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#include <istream>
#endif
namespace adl {

struct BaseView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct TupleView : std::ranges::view_base {
  std::tuple<int>* begin() const;
  std::tuple<int>* end() const;
};

struct NestedView : std::ranges::view_base {
  BaseView* begin() const;
  BaseView* end() const;
};

struct Pred {
  bool operator()(const auto&...) const;
};

struct Sentinel {
  bool operator==(const auto&) const;
};

struct Value {
  friend std::istream& operator>>(std::istream&, Value);
};

void adl_func(const auto&);

} // namespace adl

template <class View>
concept CanFindADLFunc = requires(std::ranges::iterator_t<View> it) { adl_func(it); };

static_assert(!CanFindADLFunc<std::ranges::elements_view<adl::TupleView, 0>>);
static_assert(!CanFindADLFunc<std::ranges::filter_view<adl::BaseView, adl::Pred>>);
static_assert(!CanFindADLFunc<std::ranges::iota_view<int, adl::Sentinel>>);

#ifndef TEST_HAS_NO_LOCALIZATION
static_assert(!CanFindADLFunc<std::ranges::istream_view<adl::Value>>);
#endif

static_assert(!CanFindADLFunc<std::ranges::join_view<adl::NestedView>>);

static_assert(!CanFindADLFunc<std::ranges::lazy_split_view<adl::BaseView, adl::BaseView>>);
using InnerRange =
    typename std::ranges::iterator_t<std::ranges::lazy_split_view<adl::BaseView, adl::BaseView>>::value_type;
static_assert(!CanFindADLFunc<InnerRange >);

static_assert(!CanFindADLFunc<std::ranges::split_view<adl::BaseView, adl::BaseView>>);
static_assert(!CanFindADLFunc<std::ranges::transform_view<adl::BaseView, adl::Pred>>);

#if TEST_STD_VER >= 23
static_assert(!CanFindADLFunc<std::ranges::zip_view<adl::BaseView>>);
#endif
