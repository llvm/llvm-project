// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___RANGES_COUNTED_H
#define _LIBCPP___RANGES_COUNTED_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/counted_iterator.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__memory/pointer_traits.h>
#include <__ranges/concepts.h>
#include <__ranges/subrange.h>
#include <__utility/__decay_copy.h>
#include <__utility/declval.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <span>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace views {

namespace __counted {
  template<class _From, class _To>
  concept __explicitly_convertible = requires {
    _To(_From{});
  };

  struct __fn {
    template<class _Iter, class _Diff>
      requires contiguous_iterator<decay_t<_Iter>> &&
               __explicitly_convertible<_Diff, iter_difference_t<_Iter>>
    _LIBCPP_HIDE_FROM_ABI
    constexpr auto operator()(_Iter&& __it, _Diff __c) const
      noexcept(noexcept(
        span(_VSTD::to_address(__it), static_cast<iter_difference_t<_Iter>>(__c))
      ))
    {
      return span(_VSTD::to_address(__it), static_cast<iter_difference_t<_Iter>>(__c));
    }

    template<class _Iter, class _Diff>
      requires random_access_iterator<decay_t<_Iter>> &&
               __explicitly_convertible<_Diff, iter_difference_t<_Iter>>
    _LIBCPP_HIDE_FROM_ABI
    constexpr auto operator()(_Iter&& __it, _Diff __c) const
      noexcept(
        noexcept(__it + static_cast<iter_difference_t<_Iter>>(__c)) &&
        noexcept(ranges::subrange(_VSTD::forward<_Iter>(__it), _VSTD::__decay_copy(__it)))
      )
    {
      auto __last = __it + static_cast<iter_difference_t<_Iter>>(__c);
      return ranges::subrange(_VSTD::forward<_Iter>(__it), _VSTD::move(__last));
    }

    template<class _Iter, class _Diff>
      requires __explicitly_convertible<_Diff, iter_difference_t<_Iter>>
    _LIBCPP_HIDE_FROM_ABI
    constexpr auto operator()(_Iter&& __it, _Diff __c) const
      noexcept(noexcept(
        ranges::subrange(counted_iterator(_VSTD::forward<_Iter>(__it), __c), default_sentinel)
      ))
    {
      return ranges::subrange(counted_iterator(_VSTD::forward<_Iter>(__it), __c), default_sentinel);
    }
  };
}

inline namespace __cpo {
  inline constexpr auto counted = __counted::__fn{};
} // namespace __cpo

} // namespace views

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_COUNTED_H
