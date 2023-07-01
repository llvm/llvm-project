//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_IS_POINTER_IN_RANGE_H
#define _LIBCPP___UTILITY_IS_POINTER_IN_RANGE_H

#include <__algorithm/comp.h>
#include <__assert>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Up>
_LIBCPP_CONSTEXPR_SINCE_CXX14 _LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_SANITIZE("address") bool __is_pointer_in_range(
    const _Tp* __begin, const _Tp* __end, const _Up* __ptr) {
  static_assert(!is_function<_Tp>::value && !is_function<_Up>::value,
                "__is_pointer_in_range should not be called with function pointers");
  static_assert(!is_member_pointer<_Tp>::value && !is_member_pointer<_Up>::value,
                "__is_pointer_in_range should not be called with member pointers");

  if (__libcpp_is_constant_evaluated()) {
    _LIBCPP_ASSERT_UNCATEGORIZED(__builtin_constant_p(__begin <= __end), "__begin and __end do not form a range");

    // If this is not a constant during constant evaluation we know that __ptr is not part of the allocation where
    // [__begin, __end) is.
    if (!__builtin_constant_p(__begin <= __ptr && __ptr < __end))
      return false;
  }

  // Checking this for unrelated pointers is technically UB, but no compiler optimizes based on it (currently).
  return !__less<>()(__ptr, __begin) && __less<>()(__ptr, __end);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_IS_POINTER_IN_RANGE_H
