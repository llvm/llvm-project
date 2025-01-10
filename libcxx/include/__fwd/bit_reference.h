//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_BIT_REFERENCE_H
#define _LIBCPP___FWD_BIT_REFERENCE_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_unsigned.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Cp, bool _IsConst, typename _Cp::__storage_type = 0>
class __bit_iterator;

template <class, class = void>
struct __size_difference_type_traits;

template <class _StorageType, __enable_if_t<is_unsigned<_StorageType>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _StorageType __leading_mask(unsigned __shift) {
  return static_cast<_StorageType>(static_cast<_StorageType>(~static_cast<_StorageType>(0)) << __shift);
}

template <class _StorageType, __enable_if_t<is_unsigned<_StorageType>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _StorageType __trailing_mask(unsigned __shift) {
  return static_cast<_StorageType>(static_cast<_StorageType>(~static_cast<_StorageType>(0)) >> __shift);
}

template <class _StorageType, __enable_if_t<is_unsigned<_StorageType>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _StorageType __middle_mask(unsigned __lshift, unsigned __rshift) {
  return static_cast<_StorageType>(
      std::__leading_mask<_StorageType>(__lshift) & std::__trailing_mask<_StorageType>(__rshift));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_BIT_REFERENCE_H
