//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto end() requires forward_range<View> && common_range<View>;
// constexpr auto end() const;

#include <ranges>

#include <cassert>
#include <utility>
#include "test_iterators.h"
#include "types.h"

struct ForwardViewCommonIfConst : std::ranges::view_base {
  std::string_view view_;
  constexpr explicit ForwardViewCommonIfConst() = default;
  constexpr ForwardViewCommonIfConst(const char* ptr) : view_(ptr) {}
  constexpr ForwardViewCommonIfConst(std::string_view v) : view_(v) {}
  constexpr ForwardViewCommonIfConst(ForwardViewCommonIfConst&&) = default;
  constexpr ForwardViewCommonIfConst& operator=(ForwardViewCommonIfConst&&) = default;
  constexpr ForwardViewCommonIfConst(const ForwardViewCommonIfConst&) = default;
  constexpr ForwardViewCommonIfConst& operator=(const ForwardViewCommonIfConst&) = default;
  constexpr forward_iterator<char*> begin() { return forward_iterator<char*>(nullptr); }
  constexpr std::default_sentinel_t end()  { return std::default_sentinel; }
  constexpr forward_iterator<const char*> begin() const { return forward_iterator<const char*>(view_.begin()); }
  constexpr forward_iterator<const char*> end() const { return forward_iterator<const char*>(view_.end()); }
};
bool operator==(forward_iterator<char*>, std::default_sentinel_t) { return false; }

struct ForwardViewNonCommonRange : std::ranges::view_base {
  std::string_view view_;
  constexpr explicit ForwardViewNonCommonRange() = default;
  constexpr ForwardViewNonCommonRange(const char* ptr) : view_(ptr) {}
  constexpr ForwardViewNonCommonRange(std::string_view v) : view_(v) {}
  constexpr ForwardViewNonCommonRange(ForwardViewNonCommonRange&&) = default;
  constexpr ForwardViewNonCommonRange& operator=(ForwardViewNonCommonRange&&) = default;
  constexpr ForwardViewNonCommonRange(const ForwardViewNonCommonRange&) = default;
  constexpr ForwardViewNonCommonRange& operator=(const ForwardViewNonCommonRange&) = default;
  constexpr forward_iterator<char*> begin() { return forward_iterator<char*>(nullptr); }
  constexpr std::default_sentinel_t end()  { return std::default_sentinel; }
  constexpr forward_iterator<const char*> begin() const { return forward_iterator<const char*>(view_.begin()); }
  constexpr std::default_sentinel_t end() const { return std::default_sentinel; }
};
bool operator==(forward_iterator<const char*>, std::default_sentinel_t) { return false; }

constexpr bool test() {
  // non-const: forward_range<V> && simple_view<V> && simple_view<P> -> outer-iterator<Const = true>
  // const: forward_range<V> && common_range<V> -> outer-iterator<Const = true>
  {
    using V = ForwardView;
    using P = V;

    static_assert(std::ranges::forward_range<V>);
    static_assert(std::ranges::common_range<const V>);
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<P>);

    {
      std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<V> && common_range<V> && simple_view<V> && !simple_view<P> -> outer-iterator<Const=false>
  // const: forward_range<V> && forward_range<const V> && common_range<const V> -> outer-iterator<Const = false>
  {
    using V = ForwardView;
    using P = ForwardDiffView;

    static_assert(std::ranges::forward_range<V>);
    static_assert(std::ranges::common_range<V>);
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<P>);
    static_assert(std::ranges::forward_range<const V>);
    static_assert(std::ranges::common_range<const V>);

    {
      std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<V> && !common_range<V> -> disabled
  // const: forward_range<V> && forward_range<const V> && common_range<const V> -> outer-iterator<Const = true>
  {
    using V = ForwardViewCommonIfConst;
    using P = V;

    static_assert(std::ranges::forward_range<V>);
    static_assert(!std::ranges::common_range<V>);
    static_assert(std::ranges::forward_range<const V>);
    static_assert(std::ranges::common_range<const V>);

    {
      std::ranges::lazy_split_view<V, P> v;
      auto it = v.begin();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), char&>);
    }

    {
      const std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.begin();
      static_assert(std::is_same_v<decltype(it)::iterator_concept, std::forward_iterator_tag>);
      static_assert(std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<V> && !common_range<V> -> disabled
  // const: forward_range<V> && forward_range<const V> && !common_range<const V> -> outer-iterator<Const = false>
  {
    using V = ForwardViewNonCommonRange;
    using P = V;

    static_assert(std::ranges::forward_range<V>);
    static_assert(!std::ranges::common_range<V>);
    static_assert(std::ranges::forward_range<const V>);
    static_assert(!std::ranges::common_range<const V>);

    {
      std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(std::same_as<decltype(it), std::default_sentinel_t>);
    }

    {
      const std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(std::same_as<decltype(it), std::default_sentinel_t>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
