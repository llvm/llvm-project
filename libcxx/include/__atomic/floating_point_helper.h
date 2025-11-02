//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_FLOATING_POINT_HELPER_H
#define _LIBCPP___ATOMIC_FLOATING_POINT_HELPER_H

#include <__config>
#include <__type_traits/is_floating_point.h>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool __is_fp80_long_double() {
  // Only x87-fp80 long double has 64-bit mantissa
  return __LDBL_MANT_DIG__ == 64 && std::is_same_v<_Tp, long double>;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool __has_rmw_builtin() {
  static_assert(std::is_floating_point_v<_Tp>);
#  ifndef _LIBCPP_COMPILER_CLANG_BASED
  return false;
#  else
  // The builtin __cxx_atomic_fetch_add errors during compilation for
  // long double on platforms with fp80 format.
  // For more details, see
  // lib/Sema/SemaChecking.cpp function IsAllowedValueType
  // LLVM Parser does not allow atomicrmw with x86_fp80 type.
  // if (ValType->isSpecificBuiltinType(BuiltinType::LongDouble) &&
  //    &Context.getTargetInfo().getLongDoubleFormat() ==
  //        &llvm::APFloat::x87DoubleExtended())
  // For more info
  // https://llvm.org/PR68602
  // https://reviews.llvm.org/D53965
  return !std::__is_fp80_long_double<_Tp>();
#  endif
}

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_FLOATING_POINT_HELPER_H
