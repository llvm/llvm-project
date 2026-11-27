//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define _LIBCPP___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

#include <__config>
#include <__fwd/functional.h>
#include <__type_traits/is_specialization.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline const bool __is_reference_wrapper_v = __is_specialization_v<__remove_cv_t<_Tp>, reference_wrapper>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_ENABLE_IF_H
