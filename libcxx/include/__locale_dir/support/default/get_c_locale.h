//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_GET_C_LOCALE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_GET_C_LOCALE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Get the C locale object
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_EXPORTED_FROM_ABI __locale::__locale_t __cloc();
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
#define __cloc_defined

namespace __locale {
inline __locale_t __get_c_locale() { return std::__cloc(); }
} // namespace __locale

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_GET_C_LOCALE_H
