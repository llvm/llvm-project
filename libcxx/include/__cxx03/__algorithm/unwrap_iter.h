//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_UNWRAP_ITER_H
#define _LIBCPP___CXX03___ALGORITHM_UNWRAP_ITER_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__memory/pointer_traits.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// TODO: Change the name of __unwrap_iter_impl to something more appropriate
// The job of __unwrap_iter is to remove iterator wrappers (like reverse_iterator or __wrap_iter),
// to reduce the number of template instantiations and to enable pointer-based optimizations e.g. in std::copy.
//
// Some algorithms (e.g. std::copy, but not std::sort) need to convert an
// "unwrapped" result back into the original iterator type. Doing that is the job of __rewrap_iter.

// Default case - we can't unwrap anything
template <class _Iter, bool = __libcpp_is_contiguous_iterator<_Iter>::value>
struct __unwrap_iter_impl {
  static _LIBCPP_HIDE_FROM_ABI _Iter __rewrap(_Iter, _Iter __iter) { return __iter; }
  static _LIBCPP_HIDE_FROM_ABI _Iter __unwrap(_Iter __i) _NOEXCEPT { return __i; }
};

// TODO(hardening): make sure that the following unwrapping doesn't unexpectedly turn hardened iterators into raw
// pointers.

// It's a contiguous iterator, so we can use a raw pointer instead
template <class _Iter>
struct __unwrap_iter_impl<_Iter, true> {
  using _ToAddressT = decltype(std::__to_address(std::declval<_Iter>()));

  static _LIBCPP_HIDE_FROM_ABI _Iter __rewrap(_Iter __orig_iter, _ToAddressT __unwrapped_iter) {
    return __orig_iter + (__unwrapped_iter - std::__to_address(__orig_iter));
  }

  static _LIBCPP_HIDE_FROM_ABI _ToAddressT __unwrap(_Iter __i) _NOEXCEPT { return std::__to_address(__i); }
};

template <class _Iter,
          class _Impl                                             = __unwrap_iter_impl<_Iter>,
          __enable_if_t<is_copy_constructible<_Iter>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI decltype(_Impl::__unwrap(std::declval<_Iter>())) __unwrap_iter(_Iter __i) _NOEXCEPT {
  return _Impl::__unwrap(__i);
}

template <class _OrigIter, class _Iter, class _Impl = __unwrap_iter_impl<_OrigIter> >
_LIBCPP_HIDE_FROM_ABI _OrigIter __rewrap_iter(_OrigIter __orig_iter, _Iter __iter) _NOEXCEPT {
  return _Impl::__rewrap(std::move(__orig_iter), std::move(__iter));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_UNWRAP_ITER_H
