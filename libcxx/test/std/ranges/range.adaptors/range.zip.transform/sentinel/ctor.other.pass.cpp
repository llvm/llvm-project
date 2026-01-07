//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr sentinel(sentinel<!Const> i)
//    requires Const && convertible_to<zentinel<false>, zentinel<Const>>;

#include <cassert>
#include <ranges>

#include "../types.h"

template <class T>
struct convertible_sentinel_wrapper {
  explicit convertible_sentinel_wrapper() = default;
  constexpr convertible_sentinel_wrapper(const T& it) : it_(it) {}

  template <class U>
    requires std::convertible_to<const U&, T>
  constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other) : it_(other.it_) {}

  constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other) {
    return self.it_ == other;
  }
  T it_;
};

struct NonSimpleNonCommonConvertibleView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr int* begin() { return buffer_; }
  constexpr const int* begin() const { return buffer_; }
  constexpr convertible_sentinel_wrapper<int*> end() { return convertible_sentinel_wrapper<int*>(buffer_ + size_); }
  constexpr convertible_sentinel_wrapper<const int*> end() const {
    return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

// convertible_to<zentinel<false>, zentinel<Const>>
static_assert(std::convertible_to< //
              std::ranges::sentinel_t<std::ranges::zip_view<NonSimpleNonCommonConvertibleView>>,
              std::ranges::sentinel_t<std::ranges::zip_view<NonSimpleNonCommonConvertibleView> const>>);

constexpr bool test() {
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  {
    std::ranges::zip_transform_view v{
        MakeTuple{}, NonSimpleNonCommonConvertibleView(buffer1), NonSimpleNonCommonConvertibleView(buffer2)};
    using ZipTransformView = decltype(v);
    static_assert(!std::ranges::common_range<ZipTransformView>);
    auto sent1                                            = v.end();
    std::ranges::sentinel_t<const ZipTransformView> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);

    assert(v.begin() != sent2);
    assert(std::as_const(v).begin() != sent2);
    assert(v.begin() + 4 == sent2);
    assert(std::as_const(v).begin() + 4 == sent2);

    // Cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(sent1), decltype(sent2)>);
  }

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleNonCommonConvertibleView{buffer1});
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 4 == sent1);
    assert(v.begin() + 4 == sent2);
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, NonSimpleNonCommonConvertibleView{buffer1}, std::views::iota(0));
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 4 == sent1);
    assert(v.begin() + 4 == sent2);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(
        Tie{}, NonSimpleNonCommonConvertibleView{buffer1}, SimpleCommon{buffer1}, std::ranges::single_view(2.));
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 1 == sent1);
    assert(v.begin() + 1 == sent2);
  }

  {
    // single empty range
    std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleNonCommonConvertibleView(nullptr, 0));
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), NonSimpleNonCommonConvertibleView{buffer1}, SimpleCommon{buffer1});
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer1}, std::ranges::empty_view<int>(), NonSimpleNonCommonConvertibleView{buffer1});
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer1}, NonSimpleNonCommonConvertibleView{buffer1}, std::ranges::empty_view<int>());
    auto sent1                                       = v.end();
    std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
