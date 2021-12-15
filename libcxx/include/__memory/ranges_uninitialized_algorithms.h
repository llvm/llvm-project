// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RANGES_UNINITIALIZED_ALGORITHMS_H
#define _LIBCPP___MEMORY_RANGES_UNINITIALIZED_ALGORITHMS_H

#include <__concepts/constructible.h>
#include <__config>
#include <__function_like.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/readable_traits.h>
#include <__memory/concepts.h>
#include <__memory/uninitialized_algorithms.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)
namespace ranges {

// uninitialized_default_construct

namespace __uninitialized_default_construct {

struct __fn final : private __function_like {

  constexpr explicit __fn(__tag __x) noexcept : __function_like(__x) {}

  template <__nothrow_forward_iterator _ForwardIterator,
            __nothrow_sentinel_for<_ForwardIterator> _Sentinel>
    requires default_initializable<iter_value_t<_ForwardIterator>>
  _ForwardIterator operator()(_ForwardIterator __first, _Sentinel __last) const {
    using _ValueType = remove_reference_t<iter_reference_t<_ForwardIterator>>;
    return _VSTD::__uninitialized_default_construct<_ValueType>(__first, __last);
  }

  template <__nothrow_forward_range _ForwardRange>
    requires default_initializable<range_value_t<_ForwardRange>>
  borrowed_iterator_t<_ForwardRange> operator()(_ForwardRange&& __range) const {
    return (*this)(ranges::begin(__range), ranges::end(__range));
  }

};

} // namespace __uninitialized_default_construct_ns

inline namespace __cpo {
inline constexpr auto uninitialized_default_construct =
    __uninitialized_default_construct::__fn(__function_like::__tag());
} // namespace __cpo

// uninitialized_default_construct_n

namespace __uninitialized_default_construct_n {

struct __fn final : private __function_like {

  constexpr explicit __fn(__tag __x) noexcept :
      __function_like(__x) {}

  template <__nothrow_forward_iterator _ForwardIterator>
    requires default_initializable<iter_value_t<_ForwardIterator>>
  _ForwardIterator operator()(_ForwardIterator __first,
                              iter_difference_t<_ForwardIterator> __n) const {
    using _ValueType = remove_reference_t<iter_reference_t<_ForwardIterator>>;
    return _VSTD::__uninitialized_default_construct_n<_ValueType>(__first, __n);
  }

};

} // namespace __uninitialized_default_construct_n_ns

inline namespace __cpo {
inline constexpr auto uninitialized_default_construct_n =
    __uninitialized_default_construct_n::__fn(__function_like::__tag());
} // namespace __cpo

} // namespace ranges
#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_RANGES_UNINITIALIZED_ALGORITHMS_H
