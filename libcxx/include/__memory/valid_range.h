//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_VALID_RANGE_H
#define _LIBCPP___MEMORY_VALID_RANGE_H

#include <__algorithm/comp.h>
#include <__assert>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/assume_aligned.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// A valid range as defined by the C++ Standard has the following constraints:
// - [__first, __last) is dereferenceable
// - __last is reachable from __first
// - if __first and __last are contiguous iterators, the pointers they "decay to" are correctly aligned according to the
// language rules for pointers

// This function attempts to detect invalid ranges as defined above. Specifically, it checks bullet (2). This also means
// that it doesn't return whether a range is actually valid, but only whether a range is definitely not valid.
// The checks may be extended in the future.
template <class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX14 _LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_SANITIZE("address") bool
__is_valid_range(const _Tp* __first, const _Tp* __last) {
  if (__libcpp_is_constant_evaluated()) {
    // If this is not a constant during constant evaluation, that is because __first and __last are not
    // part of the same allocation. If they are part of the same allocation, we must still make sure they
    // are ordered properly.
    return __builtin_constant_p(__first <= __last) && __first <= __last;
  }

  return !__less<>()(__last, __first);
}

// This function allows the compiler to assume that [__first, __last) is a valid range as defined above.
//
// In practice, we only add explicit assumptions for bullets (1) and (3). These assumptions allow (currently only
// clang-based compilers) to auto-vectorize algorithms that contain early returns.
template <class _Iter, class _Sent>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__assume_valid_range([[__maybe_unused__]] _Iter&& __first, [[__maybe_unused__]] _Sent&& __last) {
#if defined(_LIBCPP_CLANG_VER) && _LIBCPP_CLANG_VER >= 2300 && !defined(_LIBCPP_CXX03_LANG)
  if constexpr (__libcpp_is_contiguous_iterator<__remove_cvref_t<_Iter>>::value &&
                is_same<__remove_cvref_t<_Iter>, __remove_cvref_t<_Sent>>::value) {
    _LIBCPP_ASSERT_INTERNAL(std::__is_valid_range(std::__to_address(__first), std::__to_address(__last)),
                            "Valid range assumption does not hold");
    if (!__libcpp_is_constant_evaluated()) {
      using __value_type = typename iterator_traits<__remove_cvref_t<_Iter>>::value_type;
      __builtin_assume_dereferenceable(std::__to_address(__first), (__last - __first) * sizeof(__value_type));
      (void)std::__assume_aligned<_LIBCPP_ALIGNOF(__value_type)>(std::__to_address(__first));
      (void)std::__assume_aligned<_LIBCPP_ALIGNOF(__value_type)>(std::__to_address(__last));
    }
  }
#endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_VALID_RANGE_H
