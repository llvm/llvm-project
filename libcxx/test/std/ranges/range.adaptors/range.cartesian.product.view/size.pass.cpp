//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr auto size()       requires cartesian-product-is-sized<      First,       Vs...>;
// constexpr auto size() const requires cartesian-product-is-sized<const First, const Vs...>;

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <type_traits>

#include "test_macros.h"

#include "../range_adaptor_types.h"

template <class T>
concept HasSize = requires(T&& t) { t.size(); };

// A sized view whose range_size_t is chosen by the caller, so the common_type_t fold in
// cartesian_product_view::size() can be observed. Mirrors DiffTypeRange in
// iterator/member_types.compile.pass.cpp, which does the same for difference_type.
template <class SizeT>
struct SizeTypeView : std::ranges::view_base {
  int* buffer_;
  std::size_t size_;

  template <std::size_t N>
  constexpr SizeTypeView(std::array<int, N>& a) : buffer_(a.data()), size_(N) {}

  constexpr int* begin() const { return buffer_; }
  constexpr int* end() const { return buffer_ + size_; }
  constexpr SizeT size() const { return static_cast<SizeT>(size_); }
};
static_assert(std::same_as<std::ranges::range_size_t<SizeTypeView<unsigned short>>, unsigned short>);
static_assert(std::same_as<std::ranges::range_size_t<SizeTypeView<unsigned int>>, unsigned int>);
static_assert(std::same_as<std::ranges::range_size_t<SizeTypeView<unsigned long long>>, unsigned long long>);

constexpr bool test() {
  { // example from cppreference
    constexpr static auto w = {1};
    constexpr static auto x = {2, 3};
    constexpr static auto y = {4, 5, 6};
    constexpr static auto z = {7, 8, 9, 10, 11, 12, 13};

    constexpr auto v = std::ranges::cartesian_product_view(
        std::views::all(w), std::views::all(x), std::views::all(y), std::views::all(z));

    assert(v.size() == 42);
    assert(v.size() == w.size() * x.size() * y.size() * z.size());
  }

  { // empty range yields size 0
    std::ranges::empty_view<int> e;
    auto v = std::ranges::cartesian_product_view(e);
    assert(v.size() == 0);
  }

  { // 1-3 ranges
    constexpr std::size_t N0 = 3, N1 = 7, N2 = 42;
    std::array<int, N0> a0{};
    std::array<int, N1> a1{};
    std::array<int, N2> a2{};
    assert(std::ranges::cartesian_product_view(a0).size() == N0);
    assert(std::ranges::cartesian_product_view(a0, a1).size() == N0 * N1);
    assert(std::ranges::cartesian_product_view(a0, a1, a2).size() == N0 * N1 * N2);
  }

  { // size() return type -- common_type of underlying range_size_t
    std::array<int, 3> a;
    auto v = std::ranges::cartesian_product_view(a);
    static_assert(std::unsigned_integral<decltype(v.size())>);
  }

  { // size() folds mixed-width range_size_t values through their common type.
    // The return type is the regression guard here: dropping the static_cast, or picking the
    // first range's range_size_t instead of the common type, changes it. The *value* cannot
    // be made to differ at constexpr-feasible extents, since integral promotion widens any
    // narrow-typed multiply to int before it can wrap.
    // The exact type is implementation-defined, so only libc++ pins it.
    std::array<int, 4> buf4{};
    std::array<int, 7> buf7{};
    SizeTypeView<unsigned short> narrow{buf4};
    SizeTypeView<unsigned long long> wide{buf7};

    auto v = std::ranges::cartesian_product_view(narrow, wide);
    static_assert(std::unsigned_integral<decltype(v.size())>);
    LIBCPP_STATIC_ASSERT(
        std::same_as<decltype(v.size()), std::common_type_t<std::size_t, unsigned short, unsigned long long>>);
    LIBCPP_STATIC_ASSERT(std::same_as<decltype(v.size()), unsigned long long>);
    assert(v.size() == 28);
  }

  { // the fold runs over more than two distinct size types
    std::array<int, 2> buf2{};
    std::array<int, 3> buf3{};
    std::array<int, 5> buf5{};
    SizeTypeView<unsigned short> a{buf2};
    SizeTypeView<unsigned int> b{buf3};
    SizeTypeView<unsigned long long> c{buf5};

    auto v = std::ranges::cartesian_product_view(a, b, c);
    static_assert(std::unsigned_integral<decltype(v.size())>);
    LIBCPP_STATIC_ASSERT(
        std::same_as<decltype(v.size()),
                     std::common_type_t<std::size_t, unsigned short, unsigned int, unsigned long long>>);
    assert(v.size() == 30);
  }

  { // bases narrower than size_t do not narrow the product's size type, which is what the product
    // of all the sizes is computed in
    std::array<int, 4> buf4{};
    std::array<int, 7> buf7{};
    SizeTypeView<unsigned short> a{buf4};
    SizeTypeView<unsigned short> b{buf7};

    auto v = std::ranges::cartesian_product_view(a, b);
    static_assert(std::unsigned_integral<decltype(v.size())>);
    LIBCPP_STATIC_ASSERT(std::same_as<decltype(v.size()), std::size_t>);
    assert(v.size() == 28);
  }

  return true;
}

// Negative case: an unsized range disables size() on the cartesian product.
// (NonSizedRandomAccessView from range_adaptor_types.h is random-access but not sized.)
static_assert(!std::ranges::sized_range<NonSizedRandomAccessView>);
static_assert(!HasSize<std::ranges::cartesian_product_view<NonSizedRandomAccessView>>);
static_assert(!HasSize<std::ranges::cartesian_product_view<SimpleCommon, NonSizedRandomAccessView>>);

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
