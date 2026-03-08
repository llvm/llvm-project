//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_FUNCTION_REF_H
#define _LIBCPP___FUNCTIONAL_FUNCTION_REF_H

#include <__config>
#include <__functional/function_ref_common.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_FUNCTION_REF
// NOLINTBEGIN(readability-duplicate-include)

#define _LIBCPP_FUNCTION_REF_CV
#define _LIBCPP_FUNCTION_REF_NOEXCEPT false
#include <__functional/function_ref_impl.h>
#undef _LIBCPP_FUNCTION_REF_CV
#undef _LIBCPP_FUNCTION_REF_NOEXCEPT

#define _LIBCPP_FUNCTION_REF_CV
#define _LIBCPP_FUNCTION_REF_NOEXCEPT true
#include <__functional/function_ref_impl.h>
#undef _LIBCPP_FUNCTION_REF_CV
#undef _LIBCPP_FUNCTION_REF_NOEXCEPT

#define _LIBCPP_FUNCTION_REF_CV const
#define _LIBCPP_FUNCTION_REF_NOEXCEPT false
#include <__functional/function_ref_impl.h>
#undef _LIBCPP_FUNCTION_REF_CV
#undef _LIBCPP_FUNCTION_REF_NOEXCEPT

#define _LIBCPP_FUNCTION_REF_CV const
#define _LIBCPP_FUNCTION_REF_NOEXCEPT true
#include <__functional/function_ref_impl.h>
#undef _LIBCPP_FUNCTION_REF_CV
#undef _LIBCPP_FUNCTION_REF_NOEXCEPT

// NOLINTEND(readability-duplicate-include)
#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_FUNCTION_REF

#endif // _LIBCPP___FUNCTIONAL_FUNCTION_REF_H
