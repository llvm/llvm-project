// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_AS_CONST_VIEW_H
#define _LIBCPP___RANGES_AS_CONST_VIEW_H

#include <__concepts/constructible.h>
#include <__iterator/concepts.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/const_access.h>
#include <__ranges/empty_view.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__type_traits/is_specialization.h>
#include <__utility/auto_cast.h>
#include <__utility/move.h>
#include <cstddef>
#include <span>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {
template <input_range _View>
  requires view<_View>
class as_const_view : public view_interface<as_const_view<_View>> {
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
  _LIBCPP_HIDE_FROM_ABI as_const_view()
    requires default_initializable<_View>
  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit as_const_view(_View __base) : __base_(std::move(__base)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return ranges::cbegin(__base_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _View>
  {
    return ranges::cbegin(__base_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    return ranges::cend(__base_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _View>
  {
    return ranges::cend(__base_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return ranges::size(__base_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return ranges::size(__base_);
  }
};

template <class _Range>
as_const_view(_Range&&) -> as_const_view<views::all_t<_Range>>;

template <class _Tp>
inline constexpr bool enable_borrowed_range<as_const_view<_Tp>> = enable_borrowed_range<_Tp>;

namespace views {
namespace __as_const {

template <class _Tp>
concept __has_type = requires { typename _Tp::type; };

template <class _Tp>
struct __empty_view_case {};
template <class _Tp>
struct __empty_view_case<empty_view<_Tp>> {
  using type = const _Tp;
};

template <class _Tp>
struct __span_case {};
template <class _Tp, size_t _Extent>
struct __span_case<span<_Tp, _Extent>> {
  using type = span<const _Tp, _Extent>;
};

template <class _Tp>
struct __ref_view_case {};
template <class _Tp>
  requires constant_range<const _Tp>
struct __ref_view_case<ref_view<_Tp>> {
  using type = const _Tp&;
};

template <class _Tp>
struct __constant_range_case {};
template <class _Tp>
  requires constant_range<const _Tp> && (!view<_Tp>)
struct __constant_range_case<_Tp> {
  using type = const _Tp&;
};

struct __fn : __range_adaptor_closure<__fn> {
  // [range.as.const.overview]: the basic `constant_range` case
  template <class _Range>
    requires constant_range<all_t<_Range>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(views::all(std::forward<_Range>(__range))))
      -> decltype(/*--------------------------*/ views::all(std::forward<_Range>(__range))) {
    return /*---------------------------------*/ views::all(std::forward<_Range>(__range));
  }

  // [range.as.const.overview]: the `empty_view` case
  template <class _Range, class _UType = std::remove_cvref_t<_Range>>
    requires(!constant_range<all_t<_Range>>) && __has_type<__empty_view_case<_UType>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&&) noexcept(noexcept(auto(views::empty<typename __empty_view_case<_UType>::type>)))
      -> decltype(/*------------------*/ auto(views::empty<typename __empty_view_case<_UType>::type>)) {
    return /*-------------------------*/ auto(views::empty<typename __empty_view_case<_UType>::type>);
  }

  // [range.as.const.overview]: the `span` case
  template <class _Range, class _UType = std::remove_cvref_t<_Range>>
    requires(!constant_range<all_t<_Range>>) && __has_type<__span_case<_UType>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(typename __span_case<_UType>::type(std::forward<_UType>(__range))))
      -> decltype(/*--------------------------*/ typename __span_case<_UType>::type(std::forward<_UType>(__range))) {
    return /*---------------------------------*/ typename __span_case<_UType>::type(std::forward<_UType>(__range));
  }

  // [range.as.const.overview]: the `ref_view` case
  template <class _Range, class _UType = std::remove_cvref_t<_Range>>
    requires(!constant_range<all_t<_Range>>) && __has_type<__ref_view_case<_UType>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(ref_view(static_cast<typename __ref_view_case<_UType>::type>(__range.base()))))
      -> decltype(/*--------------------------*/ ref_view(
          static_cast<typename __ref_view_case<_UType>::type>(__range.base()))) {
    return /*---------------------------------*/ ref_view(
        static_cast<typename __ref_view_case<_UType>::type>(__range.base()));
  }

  // [range.as.const.overview]: the second `constant_range` case
  template <class _Range, class _UType = std::remove_cvref_t<_Range>>
    requires(!constant_range<all_t<_Range>>) && is_lvalue_reference_v<_Range> &&
                __has_type<__constant_range_case<_UType>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(ref_view(static_cast<typename __constant_range_case<_UType>::type>(__range))))
      -> decltype(/*--------------------------*/ ref_view(
          static_cast<typename __constant_range_case<_UType>::type>(__range))) {
    return /*---------------------------------*/ ref_view(
        static_cast<typename __constant_range_case<_UType>::type>(__range));
  }

  // [range.as.const.overview]: otherwise
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(as_const_view(std::forward<_Range>(__range))))
      -> decltype(/*--------------------------*/ as_const_view(std::forward<_Range>(__range))) {
    return /*---------------------------------*/ as_const_view(std::forward<_Range>(__range));
  }
};

} // namespace __as_const

inline namespace __cpo {
inline constexpr auto as_const = __as_const::__fn{};
} // namespace __cpo
} // namespace views

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_AS_CONST_VIEW_H
