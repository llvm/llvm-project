//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___GET_LOCALE_ENCODING_H
#define _LIBCPP___GET_LOCALE_ENCODING_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_HAS_LOCALIZATION

#  include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD
string_view _LIBCPP_EXPORTED_FROM_ABI __get_locale_encoding(const char* __name);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___GET_LOCALE_ENCODING_H
