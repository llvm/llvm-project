// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_RANGES_ITERATOR_TRAITS_H
#define _LIBCPP___ITERATOR_RANGES_ITERATOR_TRAITS_H

#include <__config>
#include <__fwd/pair.h>
#include <__ranges/concepts.h>
#include <__type_traits/remove_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

template <ranges::input_range _Range>
using __range_key_type _LIBCPP_NODEBUG = __remove_const_t<typename ranges::range_value_t<_Range>::first_type>;

template <ranges::input_range _Range>
using __range_mapped_type _LIBCPP_NODEBUG = typename ranges::range_value_t<_Range>::second_type;

template <ranges::input_range _Range>
using __range_to_alloc_type _LIBCPP_NODEBUG =
    pair<const typename ranges::range_value_t<_Range>::first_type, typename ranges::range_value_t<_Range>::second_type>;

_LIBCPP_END_NAMESPACE_STD

#endif

#endif // _LIBCPP___ITERATOR_RANGES_ITERATOR_TRAITS_H
