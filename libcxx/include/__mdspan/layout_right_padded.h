// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___MDSPAN_LAYOUT_RIGHT_PADDED_H
#define _LIBCPP___MDSPAN_LAYOUT_RIGHT_PADDED_H

#include <__assert>
#include <__config>
#include <__fwd/mdspan.h>
#include <__fwd/span.h>
#include <__mdspan/extents.h>
#include <__mdspan/layout_common.h>
#include <__mdspan/layout_stride.h>
#include <__memory/addressof.h>
#include <__type_traits/common_type.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__utility/integer_sequence.h>
#include <array>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <size_t _PaddingValue>
template <class _Extents>
class layout_right_padded<_PaddingValue>::mapping {
public:
  static_assert(__mdspan_detail::__is_extents<_Extents>::value,
                "layout_right_padded::mapping template argument must be a specialization of extents.");

  static constexpr size_t padding_value = _PaddingValue;

  using extents_type = _Extents;
  using index_type   = extents_type::index_type;
  using size_type    = extents_type::size_type;
  using rank_type    = extents_type::rank_type;
  using layout_type  = layout_right_padded;

private:
  static constexpr rank_type __rank_           = extents_type::rank();
  static constexpr size_t __last_static_extent = __rank_ == 0 ? 0uz : extents_type::static_extent(__rank_ - 1);

  static constexpr size_t __static_padding_stride =
      __mdspan_detail::__compute_static_padding_stride(__rank_, padding_value, __last_static_extent);

  // Do not store value if __static_padding_stride is not dynamic_extent.
  using __stride_rm2_type = std::extents<index_type, __static_padding_stride>;

  _LIBCPP_HIDE_FROM_ABI static constexpr bool __index_space_size_is_representable(const extents_type& __ext) {
    for (rank_type __r = 0; __r < __rank_; ++__r) {
      if (__ext.extent(__r) == static_cast<index_type>(0))
        return true;
    }

    index_type __prod = static_cast<index_type>(1);
    for (rank_type __r = 0; __r < __rank_; ++__r) {
      bool __overflowed = __builtin_mul_overflow(__prod, __ext.extent(__r), std::addressof(__prod));
      if (__overflowed)
        return false;
    }

    return true;
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr bool
  __padded_product_is_representable(const extents_type& __ext, index_type __stride_rm2) {
    if (__stride_rm2 == static_cast<index_type>(0))
      return true;
    for (rank_type __r = 0; __r < __rank_ - 1; ++__r) {
      if (__ext.extent(__r) == static_cast<index_type>(0))
        return true;
    }

    index_type __prod = __stride_rm2;
    for (rank_type __r = 0; __r < __rank_ - 1; ++__r) {
      bool __overflowed = __builtin_mul_overflow(__prod, __ext.extent(__r), std::addressof(__prod));
      if (__overflowed)
        return false;
    }
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr index_type __stride_rm2() const noexcept {
    if constexpr (__rank_ <= 1)
      return static_cast<index_type>(0);
    if constexpr (__static_padding_stride != dynamic_extent)
      return static_cast<index_type>(__static_padding_stride);
    return __stride_rm2_.extent(0);
  }

  _LIBCPP_HIDE_FROM_ABI static consteval bool __static_padded_product_is_representable() {
    if constexpr (__rank_ <= 1 || padding_value == dynamic_extent || extents_type::rank_dynamic() > 0)
      return true;
    for (rank_type __r = 0; __r < __rank_; ++__r) {
      if (extents_type::static_extent(__r) == 0)
        return true;
    }
    if constexpr (__static_padding_stride == dynamic_extent ||
                  !__mdspan_detail::__is_representable_as<index_type>(__static_padding_stride))
      return false;
    size_t __prod = __static_padding_stride;
    for (rank_type __r = 0; __r < __rank_ - 1; ++__r) {
      bool __overflowed = __builtin_mul_overflow(__prod, extents_type::static_extent(__r), std::addressof(__prod));
      if (__overflowed)
        return false;
    }
    return __mdspan_detail::__is_representable_as<index_type>(__prod);
  }

  static_assert(extents_type::rank_dynamic() != 0 || __index_space_size_is_representable(extents_type{}),
                "layout_right_padded::mapping index space for static extents must be representable as index_type.");

  static_assert(padding_value == dynamic_extent || __mdspan_detail::__is_representable_as<index_type>(padding_value),
                "layout_right_padded::mapping padding_value must be representable as index_type.");

  static_assert(__rank_ <= 1 || padding_value == dynamic_extent || __last_static_extent == dynamic_extent ||
                    (__mdspan_detail::__least_multiple_at_least_is_representable_as<size_t>(
                         padding_value, __last_static_extent) &&
                     __mdspan_detail::__least_multiple_at_least_is_representable_as<index_type>(
                         padding_value, __last_static_extent)),
                "layout_right_padded::mapping padded stride for the last static extent must be representable as "
                "size_t and index_type.");

  static_assert(__static_padded_product_is_representable(),
                "layout_right_padded::mapping required span size for static extents must be representable as size_t "
                "and index_type.");

public:
  _LIBCPP_HIDE_FROM_ABI constexpr mapping() noexcept : mapping(extents_type{}) {}
  _LIBCPP_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept = default;
  _LIBCPP_HIDE_FROM_ABI constexpr mapping(const extents_type& __ext) : __extents_(__ext) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __index_space_size_is_representable(__ext),
        "layout_right_padded::mapping(extents): index space size must be representable as index_type.");

    if constexpr (__rank_ > 1) {
      index_type __stride_rm2 = 0;
      if constexpr (padding_value == dynamic_extent) {
        __stride_rm2 = __ext.extent(__rank_ - 1);
      } else {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
            __mdspan_detail::__least_multiple_at_least_is_representable_as<index_type>(
                static_cast<index_type>(padding_value), __ext.extent(__rank_ - 1)),
            "layout_right_padded::mapping(extents): padded stride must be representable as index_type.");
        __stride_rm2 = __mdspan_detail::__least_multiple_at_least(
            static_cast<index_type>(padding_value), __ext.extent(__rank_ - 1));
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
            __padded_product_is_representable(__ext, __stride_rm2),
            "layout_right_padded::mapping(extents): required span size must be representable as index_type.");
      }
      if constexpr (__static_padding_stride == dynamic_extent)
        __stride_rm2_ = __stride_rm2_type(__stride_rm2);
    }
  }

  template <class _OtherIndexType>
    requires is_convertible_v<_OtherIndexType, index_type> && is_nothrow_constructible_v<index_type, _OtherIndexType>
  _LIBCPP_HIDE_FROM_ABI constexpr mapping(const extents_type& __ext, _OtherIndexType __padding) : __extents_(__ext) {
    auto __pad = extents_type::__index_cast(std::move(__padding));
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __mdspan_detail::__is_representable_as<index_type>(__pad),
        "layout_right_padded::mapping(extents, padding): padding must be representable as index_type.");
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __pad > 0, "layout_right_padded::mapping(extents, padding): padding must be greater than 0.");

    if constexpr (padding_value != dynamic_extent)
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          padding_value == __pad, "layout_right_padded::mapping(extents, padding): padding must equal padding_value.");

    if constexpr (__rank_ > 1) {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __mdspan_detail::__least_multiple_at_least_is_representable_as<index_type>(
              static_cast<index_type>(__pad), __ext.extent(__rank_ - 1)),
          "layout_right_padded::mapping(extents, padding): padded stride must be representable as index_type.");

      const index_type __stride_rm2 =
          __mdspan_detail::__least_multiple_at_least(static_cast<index_type>(__pad), __ext.extent(__rank_ - 1));

      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __padded_product_is_representable(__ext, __stride_rm2),
          "layout_right_padded::mapping(extents, padding): required span size must be representable as index_type.");

      if constexpr (__static_padding_stride == dynamic_extent)
        __stride_rm2_ = __stride_rm2_type(__stride_rm2);
    }
  }

  template <class _OtherExtents>
    requires is_constructible_v<extents_type, _OtherExtents>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit(!is_convertible_v<_OtherExtents, extents_type>)
      mapping(const layout_right::mapping<_OtherExtents>& __other)
      : mapping(extents_type(__other.extents())) {
    static_assert(_OtherExtents::rank() <= 1 || __static_padding_stride == dynamic_extent ||
                  _OtherExtents::static_extent(_OtherExtents::rank() - 1) == dynamic_extent ||
                  __static_padding_stride == _OtherExtents::static_extent(_OtherExtents::rank() - 1));

    if constexpr (__rank_ > 1 && padding_value != dynamic_extent) {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __other.stride(__rank_ - 2) ==
              __mdspan_detail::__least_multiple_at_least(
                  static_cast<index_type>(padding_value),
                  static_cast<index_type>(__other.extents().extent(__rank_ - 1))),
          "layout_right_padded::mapping from layout_right ctor: source stride must match the padded stride implied by "
          "padding_value.");
    }

    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
        "layout_right_padded::mapping from layout_right ctor: other.required_span_size() must be representable as "
        "index_type.");
  }

  template <class _OtherExtents>
    requires is_constructible_v<extents_type, _OtherExtents>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit(!(__rank_ == 0 && is_convertible_v<_OtherExtents, extents_type>))
      mapping(const layout_stride::mapping<_OtherExtents>& __other)
      : __extents_(extents_type(__other.extents())) {
    if constexpr (__rank_ > 1 && padding_value != dynamic_extent)
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __other.stride(__rank_ - 2) ==
              __mdspan_detail::__least_multiple_at_least(
                  static_cast<index_type>(padding_value),
                  static_cast<index_type>(__other.extents().extent(__rank_ - 1))),
          "layout_right_padded::mapping from layout_stride ctor: source stride must match the padded stride implied by "
          "padding_value.");

    if constexpr (__rank_ > 0)
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __other.stride(__rank_ - 1) == 1,
          "layout_right_padded::mapping from layout_stride ctor: stride(rank() - 1) must equal 1.");

    if constexpr (__rank_ > 2) {
      using _Common      = common_type_t<index_type, typename layout_stride::mapping<_OtherExtents>::index_type>;
      _Common __expected = static_cast<_Common>(__other.stride(__rank_ - 2));
      for (size_t __r = __rank_ - 2; __r-- > 0;) {
        __expected *= static_cast<_Common>(__other.extents().extent(__r + 1));
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
            static_cast<_Common>(__other.stride(__r)) == __expected,
            "layout_right_padded::mapping from layout_stride ctor: strides are not "
            "compatible with layout_right_padded.");
      }
    }

    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
        "layout_right_padded::mapping from layout_stride ctor: other.required_span_size() must be representable as "
        "index_type.");

    if constexpr (__rank_ > 1 && __static_padding_stride == dynamic_extent) {
#  if 0
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __mdspan_detail::__is_representable_as<index_type>(__other.stride(__rank_ - 2)),
          "layout_right_padded::mapping from layout_stride ctor: source padded stride must be representable as "
          "index_type.");
#  endif
      __stride_rm2_ = __stride_rm2_type(static_cast<index_type>(__other.stride(__rank_ - 2)));
    }
  }

  template <class _LayoutRightPaddedMapping>
    requires __mdspan_detail::__layout_right_padded_mapping_of<_LayoutRightPaddedMapping> &&
             is_constructible_v<extents_type, typename _LayoutRightPaddedMapping::extents_type>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit(
      !is_convertible_v<typename _LayoutRightPaddedMapping::extents_type, extents_type> ||
      (__rank_ > 1 && (padding_value != dynamic_extent || _LayoutRightPaddedMapping::padding_value == dynamic_extent)))
      mapping(const _LayoutRightPaddedMapping& __other)
      : __extents_(extents_type(__other.extents())) {
    static_assert(
        __rank_ <= 1 || padding_value == dynamic_extent || _LayoutRightPaddedMapping::padding_value == dynamic_extent ||
            padding_value == _LayoutRightPaddedMapping::padding_value,
        "layout_right_padded::mapping converting ctor: incompatible static padding values.");

    if constexpr (__rank_ > 1 && padding_value != dynamic_extent)
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __other.stride(__rank_ - 2) ==
              __mdspan_detail::__least_multiple_at_least(
                  static_cast<index_type>(padding_value),
                  static_cast<index_type>(__other.extents().extent(__rank_ - 1))),
          "layout_right_padded::mapping from layout_right_padded ctor: source stride must match the padded stride "
          "implied by padding_value.");

    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
        "layout_right_padded::mapping from layout_right_padded ctor: other.required_span_size() must be representable "
        "as index_type.");

    if constexpr (__rank_ > 1 && __static_padding_stride == dynamic_extent) {
#  if 0
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __mdspan_detail::__is_representable_as<index_type>(__other.stride(__rank_ - 2)),
          "layout_right_padded::mapping from layout_right_padded ctor: source padded stride must be representable as "
          "index_type.");
#  endif
      __stride_rm2_ = __stride_rm2_type(static_cast<index_type>(__other.stride(__rank_ - 2)));
    }
  }

  template <class _LayoutLeftPaddedMapping>
    requires(__mdspan_detail::__layout_left_padded_mapping_of<_LayoutLeftPaddedMapping> ||
             __mdspan_detail::__layout_left_mapping_of<_LayoutLeftPaddedMapping>) &&
            (__rank_ <= 1) && is_constructible_v<extents_type, typename _LayoutLeftPaddedMapping::extents_type>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit(
      !is_convertible_v<typename _LayoutLeftPaddedMapping::extents_type, extents_type>)
      mapping(const _LayoutLeftPaddedMapping& __other) noexcept
      : __extents_(extents_type(__other.extents())) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
        "layout_right_padded::mapping converting ctor: other.required_span_size() must be representable as "
        "index_type.");
  }

  _LIBCPP_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept { return __extents_; }
  _LIBCPP_HIDE_FROM_ABI constexpr array<index_type, __rank_> strides() const noexcept {
    return [&]<size_t... _Pos>(index_sequence<_Pos...>) {
      return array<index_type, __rank_>{stride(_Pos)...};
    }(make_index_sequence<__rank_>());
  }

  _LIBCPP_HIDE_FROM_ABI constexpr index_type required_span_size() const noexcept {
    return [&]<size_t... _Pos>(index_sequence<_Pos...>) {
      if (((__extents_.extent(_Pos) == static_cast<index_type>(0)) || ...))
        return static_cast<index_type>(0);

      return static_cast<index_type>(
          (*this)(static_cast<index_type>(__extents_.extent(_Pos) - static_cast<index_type>(1))...) +
          static_cast<index_type>(1));
    }(make_index_sequence<__rank_>());
  }

  template <class... _Indices>
    requires(sizeof...(_Indices) == __rank_) && (is_convertible_v<_Indices, index_type> && ...) &&
            (is_nothrow_constructible_v<index_type, _Indices> && ...)
  _LIBCPP_HIDE_FROM_ABI constexpr index_type operator()(_Indices... __idx) const noexcept {
    return [&]<class... _IndexTypes>(_IndexTypes... __idxs) {
      _LIBCPP_ASSERT_UNCATEGORIZED(__mdspan_detail::__is_multidimensional_index_in(__extents_, __idxs...),
                                   "layout_right_padded::mapping: out of bounds indexing.");
      return [&]<size_t... _Pos>(index_sequence<_Pos...>) {
        return ((static_cast<index_type>(__idxs) * stride(_Pos)) + ... + static_cast<index_type>(0));
      }(make_index_sequence<sizeof...(_Indices)>());
    }(extents_type::__index_cast(std::move(__idx))...);
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr bool is_always_unique() noexcept { return true; }
  _LIBCPP_HIDE_FROM_ABI static constexpr bool is_always_exhaustive() noexcept {
    if constexpr (__rank_ <= 1)
      return true;
    if constexpr (__last_static_extent == dynamic_extent || __static_padding_stride == dynamic_extent)
      return false;
    return __last_static_extent == __static_padding_stride;
  }
  _LIBCPP_HIDE_FROM_ABI static constexpr bool is_always_strided() noexcept { return true; }
  _LIBCPP_HIDE_FROM_ABI static constexpr bool is_unique() noexcept { return true; }
  _LIBCPP_HIDE_FROM_ABI constexpr bool is_exhaustive() const noexcept {
    if constexpr (__rank_ <= 1)
      return true;
    return extents().extent(__rank_ - 1) == stride(__rank_ - 2);
  }
  _LIBCPP_HIDE_FROM_ABI static constexpr bool is_strided() noexcept { return true; }

  _LIBCPP_HIDE_FROM_ABI constexpr index_type stride(rank_type __r) const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__r < __rank_, "layout_right_padded::mapping::stride(): invalid rank index.");

    if constexpr (__rank_ == 0)
      return static_cast<index_type>(1);

    if (__r == __rank_ - 1)
      return static_cast<index_type>(1);

    if constexpr (__rank_ > 1) {
      if (__r == __rank_ - 2)
        return __stride_rm2();

      index_type __stride = __stride_rm2();
      for (rank_type __i = __rank_ - 2; __i > __r; --__i)
        __stride *= __extents_.extent(__i);
      return __stride;
    }

    return static_cast<index_type>(1);
  }

  template <class _LayoutRightPaddedMapping>
    requires __mdspan_detail::__layout_right_padded_mapping_of<_LayoutRightPaddedMapping> &&
             (_LayoutRightPaddedMapping::extents_type::rank() == __rank_)
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const mapping& __x, const _LayoutRightPaddedMapping& __y) noexcept {
    return __x.extents() == __y.extents() && (__rank_ < 2 || __x.stride(__rank_ - 2) == __y.stride(__rank_ - 2));
  }

private:
  _LIBCPP_NO_UNIQUE_ADDRESS __stride_rm2_type __stride_rm2_{};
  _LIBCPP_NO_UNIQUE_ADDRESS extents_type __extents_{};
};

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MDSPAN_LAYOUT_RIGHT_PADDED_H
