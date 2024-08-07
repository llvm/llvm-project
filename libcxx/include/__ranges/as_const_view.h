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
#include <__ranges/ref_view.h>
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

template <class _Case>
concept __case = requires { _Case::__impl; };

template <class _Range>
struct __already_constant_case {};
template <class _Range>
  requires constant_range<all_t<_Range>>
struct __already_constant_case<_Range> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __impl(_Range& __range) noexcept(
      noexcept(views::all(std::forward<_Range>(__range)))) -> decltype(views::all(std::forward<_Range>(__range))) {
    return views::all(std::forward<_Range>(__range));
  }
};

template <class _Range, class _UType = std::remove_cvref_t<_Range>>
struct __empty_view_case {};
template <class _Range, class _XType>
  requires(!__case<__already_constant_case<_Range>>)
struct __empty_view_case<_Range, empty_view<_XType>> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto
  __impl(_Range&) noexcept(noexcept(auto(views::empty<const _XType>))) -> decltype(auto(views::empty<const _XType>)) {
    return auto(views::empty<const _XType>);
  }
};

template <class _Range, class _UType = std::remove_cvref_t<_Range>>
struct __span_case {};
template <class _Range, class _XType, size_t _Extent>
  requires(!__case<__already_constant_case<_Range>>)
struct __span_case<_Range, span<_XType, _Extent>> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __impl(_Range& __range) noexcept(noexcept(span<const _XType, _Extent>(
      std::forward<_Range>(__range)))) -> decltype(span<const _XType, _Extent>(std::forward<_Range>(__range))) {
    return span<const _XType, _Extent>(std::forward<_Range>(__range));
  }
};

template <class _Range, class _UType = std::remove_cvref_t<_Range>>
struct __ref_view_case {};
template <class _Range, class _XType>
  requires(!__case<__already_constant_case<_Range>>) && constant_range<const _XType>
struct __ref_view_case<_Range, ref_view<_XType>> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto
  __impl(_Range& __range) noexcept(noexcept(ref_view(static_cast<const _XType&>(std::forward<_Range>(__range).base()))))
      -> decltype(ref_view(static_cast<const _XType&>(std::forward<_Range>(__range).base()))) {
    return ref_view(static_cast<const _XType&>(std::forward<_Range>(__range).base()));
  }
};

template <class _Range, class _UType = std::remove_cvref_t<_Range>>
struct __constant_range_case {};
template <class _Range, class _UType>
  requires(!__case<__already_constant_case<_Range>>) && (!__case<__empty_view_case<_Range>>) &&
          (!__case<__span_case<_Range>>) &&
          (!__case<__ref_view_case<_Range>>) && is_lvalue_reference_v<_Range> && constant_range<const _UType> &&
          (!view<_UType>)
struct __constant_range_case<_Range, _UType> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto
  __impl(_Range& __range) noexcept(noexcept(ref_view(static_cast<const _UType&>(std::forward<_Range>(__range)))))
      -> decltype(ref_view(static_cast<const _UType&>(std::forward<_Range>(__range)))) {
    return ref_view(static_cast<const _UType&>(std::forward<_Range>(__range)));
  }
};

template <class _Range>
struct __otherwise_case {};
template <class _Range>
  requires(!__case<__already_constant_case<_Range>>) && (!__case<__empty_view_case<_Range>>) &&
          (!__case<__span_case<_Range>>) && (!__case<__ref_view_case<_Range>>) &&
          (!__case<__constant_range_case<_Range>>)
struct __otherwise_case<_Range> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __impl(_Range& __range) noexcept(noexcept(
      as_const_view(std::forward<_Range>(__range)))) -> decltype(as_const_view(std::forward<_Range>(__range))) {
    return as_const_view(std::forward<_Range>(__range));
  }
};

struct __fn : __range_adaptor_closure<__fn> {
  // [range.as.const.overview]: the basic `constant_range` case
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(noexcept(
      __already_constant_case<_Range>::__impl(__range))) -> decltype(__already_constant_case<_Range>::__impl(__range)) {
    return __already_constant_case<_Range>::__impl(__range);
  }

  // [range.as.const.overview]: the `empty_view` case
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(__empty_view_case<_Range>::__impl(__range))) -> decltype(__empty_view_case<_Range>::__impl(__range)) {
    return __empty_view_case<_Range>::__impl(__range);
  }

  // [range.as.const.overview]: the `span` case
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(__span_case<_Range>::__impl(__range))) -> decltype(__span_case<_Range>::__impl(__range)) {
    return __span_case<_Range>::__impl(__range);
  }

  // [range.as.const.overview]: the `ref_view` case
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(__ref_view_case<_Range>::__impl(__range))) -> decltype(__ref_view_case<_Range>::__impl(__range)) {
    return __ref_view_case<_Range>::__impl(__range);
  }

  // [range.as.const.overview]: the second `constant_range` case
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(noexcept(
      __constant_range_case<_Range>::__impl(__range))) -> decltype(__constant_range_case<_Range>::__impl(__range)) {
    return __constant_range_case<_Range>::__impl(__range);
  }

  // [range.as.const.overview]: otherwise
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range) noexcept(
      noexcept(__otherwise_case<_Range>::__impl(__range))) -> decltype(__otherwise_case<_Range>::__impl(__range)) {
    return __otherwise_case<_Range>::__impl(__range);
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
