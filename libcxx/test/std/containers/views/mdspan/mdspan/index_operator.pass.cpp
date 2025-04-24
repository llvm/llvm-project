//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// Test default iteration:
//
// template<class... Indices>
//   constexpr reference operator[](Indices...) const noexcept;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Preconditions:
//   * extents_type::index-cast(i) is a multidimensional index in extents_.

// GCC warns about comma operator changing its meaning inside [] in C++23
#if defined(__GNUC__) && !defined(__clang_major__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wcomma-subscript"
#endif

#include <mdspan>
#include <cassert>
#include <cstdint>
#include <span> // dynamic_extent

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"

// Apple Clang does not support argument packs as input to operator []
#ifdef TEST_COMPILER_APPLE_CLANG
template <class MDS>
constexpr auto& access(MDS mds) {
  return mds[];
}
template <class MDS>
constexpr auto& access(MDS mds, int64_t i0) {
  return mds[i0];
}
template <class MDS>
constexpr auto& access(MDS mds, int64_t i0, int64_t i1) {
  return mds[i0, i1];
}
template <class MDS>
constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2) {
  return mds[i0, i1, i2];
}
template <class MDS>
constexpr auto& access(MDS mds, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
  return mds[i0, i1, i2, i3];
}
#endif

template <class MDS, class... Indices>
concept operator_constraints = requires(MDS m, Indices... idxs) {
  { std::is_same_v<decltype(m[idxs...]), typename MDS::reference> };
};

template <class MDS, class... Indices>
  requires(operator_constraints<MDS, Indices...>)
constexpr bool check_operator_constraints(MDS m, Indices... idxs) {
  (void)m[idxs...];
  return true;
}

template <class MDS, class... Indices>
constexpr bool check_operator_constraints(MDS, Indices...) {
  return false;
}

template <class MDS, class... Args>
constexpr void iterate(MDS mds, Args... args) {
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  if constexpr (-1 == r) {
#ifdef TEST_COMPILER_APPLE_CLANG
    int* ptr1 = &access(mds, args...);
#else
    int* ptr1 = &mds[args...];
#endif
    int* ptr2 = &(mds.accessor().access(mds.data_handle(), mds.mapping()(args...)));
    assert(ptr1 == ptr2);

    std::array<typename MDS::index_type, MDS::rank()> args_arr{static_cast<typename MDS::index_type>(args)...};
    int* ptr3 = &mds[args_arr];
    assert(ptr3 == ptr2);
    int* ptr4 = &mds[std::span(args_arr)];
    assert(ptr4 == ptr2);
  } else {
    for (typename MDS::index_type i = 0; i < mds.extents().extent(r); i++) {
      iterate(mds, i, args...);
    }
  }
}

template <class Mapping>
constexpr void test_iteration(Mapping m) {
  std::array<int, 1024> data;
  using MDS = std::mdspan<int, typename Mapping::extents_type, typename Mapping::layout_type>;
  MDS mds(data.data(), m);

  iterate(mds);
}

template <class Layout>
constexpr void test_layout() {
  constexpr size_t D = std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), std::extents<int>()));
  test_iteration(construct_mapping(Layout(), std::extents<unsigned, D>(1)));
  test_iteration(construct_mapping(Layout(), std::extents<unsigned, D>(7)));
  test_iteration(construct_mapping(Layout(), std::extents<unsigned, 7>()));
  test_iteration(construct_mapping(Layout(), std::extents<unsigned, 7, 8>()));
  test_iteration(construct_mapping(Layout(), std::extents<signed char, D, D, D, D>(1, 1, 1, 1)));

// TODO(LLVM 20): Enable this once AppleClang is upgraded
#ifndef TEST_COMPILER_APPLE_CLANG
  int data[1];
  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), 0));
  static_assert(
      !check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), 0, 0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(
      check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), IntType(0)));
  static_assert(!check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<unsigned, D>(1))), IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<unsigned char, D>(1))), IntType(0)));

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not accept IntType
  static_assert(check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<unsigned char, D, D>(1, 1))), int(0), size_t(0)));
  static_assert(check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))), unsigned(0), IntType(0)));

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))),
      unsigned(0),
      IntConfig<o, o, t, t>(0)));
  static_assert(check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))),
      unsigned(0),
      IntConfig<o, t, t, t>(0)));
  static_assert(check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))),
      unsigned(0),
      IntConfig<o, t, o, t>(0)));
  static_assert(!check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))),
      unsigned(0),
      IntConfig<t, o, o, t>(0)));
  static_assert(check_operator_constraints(
      std::mdspan(data, construct_mapping(Layout(), std::extents<int, D, D>(1, 1))),
      unsigned(0),
      IntConfig<t, o, t, o>(0)));

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from const&, no-throw-ctor from non-const
  if constexpr (std::is_same_v<Layout, std::layout_left>) {
    static_assert(!check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<o, o, t, t>(0)}));
    static_assert(!check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<o, t, t, t>(0)}));
    static_assert(!check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<t, o, o, t>(0)}));
    static_assert(!check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<t, t, o, t>(0)}));
    static_assert(check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<t, o, t, o>(0)}));
    static_assert(check_operator_constraints(
        std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), std::array{IntConfig<t, t, t, t>(0)}));

    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
    {
      std::array idx{IntConfig<o, o, t, t>(0)};
      std::span s(idx);
      assert(!check_operator_constraints(std::mdspan(data, construct_mapping(Layout(), std::extents<int, D>(1))), s));
    }
  }
#endif // TEST_COMPILER_APPLE_CLANG
}

template <class Layout>
constexpr void test_layout_large() {
  constexpr size_t D = std::dynamic_extent;
  test_iteration(construct_mapping(Layout(), std::extents<int64_t, D, 4, D, D>(3, 5, 6)));
  test_iteration(construct_mapping(Layout(), std::extents<int64_t, D, 4, 1, D>(3, 6)));
}

// mdspan::operator[] casts to index_type before calling mapping
// mapping requirements only require the index operator to mixed integer types not anything convertible to index_type
constexpr void test_index_cast_happens() {}

constexpr bool test() {
  test_layout<std::layout_left>();
  test_layout<std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
  return true;
}

constexpr bool test_large() {
  test_layout_large<std::layout_left>();
  test_layout_large<std::layout_right>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  test_large();
  return 0;
}
#if defined(__GNUC__) && !defined(__clang_major__)
#  pragma GCC diagnostic pop
#endif
