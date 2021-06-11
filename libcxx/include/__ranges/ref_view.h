// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___RANGES_REF_VIEW_H
#define _LIBCPP___RANGES_REF_VIEW_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/access.h>
#include <__ranges/data.h>
#include <__ranges/view_interface.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {
  template<range _Range>
    requires is_object_v<_Range>
  class ref_view : public view_interface<ref_view<_Range>> {
    _Range *__range_;

    static void __fun(_Range&);
    static void __fun(_Range&&) = delete;

public:
    template<class _Tp>
      requires __different_from<_Tp, ref_view> &&
        convertible_to<_Tp, _Range&> && requires { __fun(declval<_Tp>()); }
    constexpr ref_view(_Tp&& __t)
      : __range_(_VSTD::addressof(static_cast<_Range&>(_VSTD::forward<_Tp>(__t))))
    {}

    constexpr _Range& base() const { return *__range_; }

    constexpr iterator_t<_Range> begin() const { return ranges::begin(*__range_); }
    constexpr sentinel_t<_Range> end() const { return ranges::end(*__range_); }

    constexpr bool empty() const
      requires requires { ranges::empty(*__range_); }
    { return ranges::empty(*__range_); }

    constexpr auto size() const
      requires sized_range<_Range>
    { return ranges::size(*__range_); }

    // TODO: This needs to use contiguous_range.
    constexpr auto data() const
      requires contiguous_iterator<iterator_t<_Range>>
    { return ranges::data(*__range_); }
  };

  template<class _Range>
  ref_view(_Range&) -> ref_view<_Range>;

} // namespace ranges

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_REF_VIEW_H
