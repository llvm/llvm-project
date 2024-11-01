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
#include <__cstddef/size_t.h>
#include <__fwd/span.h>
#include <__iterator/concepts.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/const_access.h>
#include <__ranges/empty_view.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/ref_view.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_specialization.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/auto_cast.h>
#include <__utility/declval.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <__utility/pair.h>

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
inline constexpr bool __is_span_v = false; // true if and only if _Tp is a specialization of span
template <class _Tp, size_t _Extent>
inline constexpr bool __is_span_v<span<_Tp, _Extent>> = true;

template <class _UType>
struct __xtype {
  using type = void;
};
template <class _XType>
struct __xtype<empty_view<_XType>> {
  using type = _XType;
};
template <class _XType, size_t _Extent>
struct __xtype<span<_XType, _Extent>> {
  using type                       = _XType;
  constexpr static size_t __extent = _Extent;
};
template <class _XType>
struct __xtype<ref_view<_XType>> {
  using type = _XType;
};

struct __fn : __range_adaptor_closure<__fn> {
  // implementation strategy taken from Microsoft's STL
  enum class __strategy {
    __already_const,
    __empty_view,
    __span,
    __ref_view,
    __const_is_constant_range,
    __otherwise,
    __none,
  };

  template <class _Type>
  static consteval pair<__strategy, bool> __choose_strategy() {
    using _UType = std::remove_cvref_t<_Type>;
    using _XType = __xtype<_UType>::type;

    if constexpr (!requires { typename all_t<_Type>; }) {
      return {__strategy::__none, false};
    } else if constexpr (constant_range<all_t<_Type>>) {
      return {__strategy::__already_const, noexcept(views::all(std::declval<_Type>()))};
    } else if constexpr (__is_specialization_v<_UType, empty_view>) {
      return {__strategy::__empty_view, true};
    } else if constexpr (__is_span_v<_UType>) {
      return {__strategy::__span, true};
    } else if constexpr (__is_specialization_v<_UType, ref_view> && constant_range<const _XType>) {
      return {__strategy::__ref_view, noexcept(ref_view(static_cast<const _XType&>(std::declval<_Type>().base())))};
    } else if constexpr (is_lvalue_reference_v<_Type> && constant_range<const _UType> && !view<_UType>) {
      return {__strategy::__const_is_constant_range,
              noexcept(ref_view(static_cast<const _UType&>(std::declval<_Type>())))};
    } else {
      return {__strategy::__otherwise, noexcept(as_const_view(std::declval<_Type>()))};
    }
  }

  template <class _Type>
    requires(__choose_strategy<_Type>().first != __strategy::__none)
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto
  operator()(_Type&& __range) noexcept(__choose_strategy<_Type>().second) {
    using _UType = std::remove_cvref_t<_Type>;
    using _XType = __xtype<_UType>::type;

    constexpr auto __st = __choose_strategy<_Type>().first;

    if constexpr (__st == __strategy::__already_const) {
      return views::all(std::forward<_Type>(__range));
    } else if constexpr (__st == __strategy::__empty_view) {
      return auto(views::empty<const _XType>);
    } else if constexpr (__st == __strategy::__span) {
      return span<const _XType, __xtype<_UType>::__extent>(std::forward<_Type>(__range));
    } else if constexpr (__st == __strategy::__ref_view) {
      return ref_view(static_cast<const _XType&>(std::forward<_Type>(__range).base()));
    } else if constexpr (__st == __strategy::__const_is_constant_range) {
      return ref_view(static_cast<const _UType&>(std::forward<_Type>(__range)));
    } else if constexpr (__st == __strategy::__otherwise) {
      return as_const_view(std::forward<_Type>(__range));
    }
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
