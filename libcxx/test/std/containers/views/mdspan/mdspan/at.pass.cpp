//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class... OtherIndexTypes>
//   constexpr reference at(OtherIndexTypes... indices) const;
//
// Constraints:
//   - (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   - (is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) is true, and
//   - sizeof...(OtherIndexTypes) == rank() is true.
//
// template<class OtherIndexType>
//   constexpr reference at(span<OtherIndexType, rank()> indices) const;
//
// template<class OtherIndexType>
//   constexpr reference at(const array<OtherIndexType, rank()>& indices) const;
//
// Constraints:
//   - is_convertible_v<const OtherIndexType&, index_type> is true, and
//   - is_nothrow_constructible_v<index_type, const OtherIndexType&> is true.
//
// Throws:
//   - std::out_of_range if extents_type::index-cast(indices) is not a multidimensional index in extents_.

#include <array>
#include <cassert>
#include <mdspan>
#include <span>
#include <string_view>
#include <vector>

#include "assert_macros.h"

#include "../ConvertibleToIntegral.h"
#include "../CustomTestLayouts.h"

struct SpyIndex {
  int val;
  constexpr SpyIndex(int v) : val(v) {}
  constexpr operator int() const noexcept { return val; }
};

struct CvIndex {
  constexpr operator int() noexcept { return 4; }
  constexpr operator int() const noexcept { return 3; }
};

class strict_cast_layout {
public:
  template <class Extents>
  class mapping;
};

template <class Extents>
class strict_cast_layout::mapping {
public:
  using extents_type = Extents;
  using index_type   = extents_type::index_type;
  using size_type    = extents_type::size_type;
  using rank_type    = extents_type::rank_type;
  using layout_type  = strict_cast_layout;

  constexpr mapping() noexcept               = default;
  constexpr mapping(const mapping&) noexcept = default;
  constexpr mapping(const extents_type& ext) noexcept : extents_(ext) {}

  template <class OtherExtents>
  constexpr mapping(const mapping<OtherExtents>& other) noexcept : extents_(other.extents()) {}

  constexpr mapping& operator=(const mapping&) noexcept = default;

  constexpr const extents_type& extents() const noexcept { return extents_; }

  constexpr index_type required_span_size() const noexcept {
    index_type size = 1;
    for (rank_type r = 0; r < extents_type::rank(); ++r)
      size *= extents_.extent(r);
    return size;
  }

  template <std::integral... Indices>
    requires(sizeof...(Indices) == extents_type::rank())
  constexpr index_type operator()(Indices... idx) const noexcept {
    return [&]<size_t... _Is>(std::index_sequence<_Is...>) {
      index_type res = 0;
      std::array<index_type, sizeof...(Indices)> idx_arr{static_cast<index_type>(idx)...};
      ((res = res * extents_.extent(_Is) + idx_arr[_Is]), ...);
      return res;
    }(std::make_index_sequence<sizeof...(Indices)>{});
  }

  constexpr index_type operator()(SpyIndex) const noexcept = delete;

  static constexpr bool is_always_unique() noexcept { return true; }
  static constexpr bool is_always_exhaustive() noexcept { return true; }
  static constexpr bool is_always_strided() noexcept { return true; }

  static constexpr bool is_unique() noexcept { return true; }
  static constexpr bool is_exhaustive() noexcept { return true; }
  static constexpr bool is_strided() noexcept { return true; }

  constexpr index_type stride(rank_type r) const noexcept {
    index_type stride = 1;
    for (rank_type i = r + 1; i < extents_type::rank(); ++i)
      stride *= extents_.extent(i);
    return stride;
  }

  friend constexpr bool operator==(const mapping& lhs, const mapping& rhs) noexcept {
    return lhs.extents_ == rhs.extents_;
  }

private:
  extents_type extents_{};
};

template <class MDS, class... Indices>
concept at_constraints = requires(MDS m, Indices... idxs) {
  { m.at(idxs...) } -> std::same_as<typename MDS::reference>;
};

template <class MDS, class... Indices>
  requires(at_constraints<MDS, Indices...>)
constexpr bool check_at_constraints(MDS m, Indices... idxs) {
  TEST_IGNORE_NODISCARD m.at(idxs...);
  return true;
}

template <class MDS, class... Indices>
constexpr bool check_at_constraints(MDS, Indices...) {
  return false;
}

template <class MDS, class... Args>
constexpr void iterate(MDS mds, Args... args) {
  constexpr int r = static_cast<int>(MDS::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));

  if constexpr (r == -1) {
    std::same_as<typename MDS::reference> decltype(auto) ptr_accessor =
        mds.accessor().access(mds.data_handle(), mds.mapping()(args...));
    std::array<typename MDS::index_type, MDS::rank()> args_arr{static_cast<MDS::index_type>(args)...};

    // mdspan.at(indices...)
    std::same_as<typename MDS::reference> decltype(auto) ptr_at = mds.at(args...);
    assert(&ptr_at == &ptr_accessor);

    //  mdspan.at(array)
    std::same_as<typename MDS::reference> decltype(auto) ptr_arr = mds.at(args_arr);
    assert(&ptr_arr == &ptr_accessor);

    // mdspan.at(span)
    std::same_as<typename MDS::reference> decltype(auto) ptr_span = mds.at(std::span(args_arr));
    assert(&ptr_span == &ptr_accessor);

  } else {
    for (typename MDS::index_type i = 0; i < mds.extents().extent(r); ++i) {
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
  test_iteration(construct_mapping(Layout{}, std::extents<int>{}));
  test_iteration(construct_mapping(Layout{}, std::dextents<unsigned, 1>{1}));
  test_iteration(construct_mapping(Layout{}, std::dextents<unsigned, 1>{7}));
  test_iteration(construct_mapping(Layout{}, std::extents<unsigned, 7>{}));
  test_iteration(construct_mapping(Layout{}, std::extents<unsigned, 7, 8>{}));
  test_iteration(construct_mapping(Layout{}, std::dextents<signed char, 4>{1, 1, 1, 1}));

  int data[1];
  // Check at constraint for number of arguments
  static_assert(check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, 0));
  static_assert(!check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, 0, 0));

  // Check at constraint for convertibility of arguments to index_type
  static_assert(
      check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, IntType{0}));
  static_assert(
      !check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<unsigned, 1>{1})}, IntType{0}));

  // Check at constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<unsigned char, 1>{1})}, IntType{0}));

  // Check that mixed integrals work: note the second one tests that mdspan casts: layout_wrapping_integral does not accept IntType
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<unsigned char, 2>{1, 1})}, 0, 0uz));
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0u, IntType{0}));

  constexpr bool t = true;
  constexpr bool o = false;
  static_assert(!check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<o, o, t, t>{0}));
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<o, t, t, t>{0}));
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<o, t, o, t>{0}));
  static_assert(!check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<t, o, o, t>{0}));
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<t, o, t, o>{0}));
  static_assert(check_at_constraints(
      std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 2>{1, 1})}, 0uz, IntConfig<t, t, t, t>{0}));

  // layout_wrapped wouldn't quite work here the way we wrote the check
  // IntConfig has configurable conversion properties: convert from const&, convert from non-const, no-throw-ctor from const&, no-throw-ctor from non-const
  if constexpr (std::is_same_v<Layout, std::layout_left>) {
    static_assert(!check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                        std::array{IntConfig<o, o, t, t>{0}}));
    static_assert(!check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                        std::array{IntConfig<o, t, t, t>{0}}));
    static_assert(!check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                        std::array{IntConfig<t, o, o, t>{0}}));
    static_assert(!check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                        std::array{IntConfig<t, t, o, t>{0}}));
    static_assert(check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                       std::array{IntConfig<t, o, t, o>{0}}));
    static_assert(check_at_constraints(std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})},
                                       std::array{IntConfig<t, t, t, t>{0}}));

    {
      std::array idx{IntConfig<o, o, t, t>{0}};
      assert(!check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
    {
      std::array idx{IntConfig<o, t, t, t>{0}};
      assert(!check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
    {
      std::array idx{IntConfig<t, o, o, t>{0}};
      assert(!check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
    {
      std::array idx{IntConfig<t, t, o, t>{0}};
      assert(!check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
    {
      std::array idx{IntConfig<t, o, t, o>{0}};
      assert(check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
    {
      std::array idx{IntConfig<t, t, t, t>{0}};
      assert(check_at_constraints(
          std::mdspan{data, construct_mapping(Layout{}, std::dextents<int, 1>{1})}, std::span{idx}));
    }
  }
}

constexpr bool test() {
  test_layout<std::layout_left>();
  test_layout<std::layout_right>();
  test_layout<layout_wrapping_integral<4>>();
  return true;
}

constexpr bool test_cast() {
  using MDS = std::mdspan<int, std::dextents<int, 1>, strict_cast_layout>;

  std::array data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const MDS m{data.data(), std::dextents<int, 1>{10}};

  SpyIndex index(3);
  std::array indices{index};

  assert(&m.at(index) == &data[3]);
  assert(&m.at(indices) == &data[3]);
  assert(&m.at(std::span{indices}) == &data[3]);

  std::array cv_indices{CvIndex{}};
  assert(&m.at(CvIndex{}) == &data[4]);
  assert(&m.at(cv_indices) == &data[3]);
  assert(&m.at(std::span{cv_indices}) == &data[3]);

  return true;
}

void test_throws() {
  std::array<int, 100> data{};
  std::mdspan m(data.data(), 10, 10);

  assert(&m.at(9, 9) == &data[99]);

  TEST_THROWS_TYPE(std::out_of_range, m.at(10, 0));
  TEST_THROWS_TYPE(std::out_of_range, m.at(0, 10));
  TEST_THROWS_TYPE(std::out_of_range, m.at(-1, 0));

  [[maybe_unused]] std::array bad_row{10, 0};
  TEST_THROWS_TYPE(std::out_of_range, m.at(bad_row));

  [[maybe_unused]] std::array bad_col{0, 10};
  TEST_THROWS_TYPE(std::out_of_range, m.at(std::span{bad_col}));

#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    auto verify_exception_message = [](auto&& f) {
      try {
        f();
        assert(false && "Unexpected");
      } catch (const std::out_of_range& e [[maybe_unused]]) {
        LIBCPP_ASSERT(std::string_view(e.what()) == "mdspan");
      } catch (...) {
        assert(false && "Unexpected");
      }
    };
    float arr[1024];
    // value out of range
    {
      std::mdspan mds(arr, std::extents<unsigned char, 5>());
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-130); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(256); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(1000); });
    }
    {
      std::mdspan mds(arr, std::extents<signed char, 5>());
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-130); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(128); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(1000); });
    }
    {
      std::mdspan mds(arr, std::dextents<unsigned char, 1>(5));
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-130); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(256); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(1000); });
    }
    {
      std::mdspan mds(arr, std::dextents<signed char, 1>(5));
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-130); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(128); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(1000); });
    }
    {
      std::mdspan mds(arr, std::dextents<int, 3>(5, 7, 9));
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1, -1, -1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1, 0, 0); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(0, -1, 0); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(0, 0, -1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5, 3, 3); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(3, 7, 3); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(3, 3, 9); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5, 7, 9); });
    }
    {
      std::mdspan mds(arr, std::dextents<unsigned, 3>(5, 7, 9));
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1, -1, -1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(-1, 0, 0); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(0, -1, 0); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(0, 0, -1); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5, 3, 3); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(3, 7, 3); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(3, 3, 9); });
      verify_exception_message([&] { TEST_IGNORE_NODISCARD mds.at(5, 7, 9); });
    }
  }
#endif
}

int main(int, char**) {
  test();
  static_assert(test());

  test_cast();
  static_assert(test_cast());

  test_throws();

  return 0;
}
