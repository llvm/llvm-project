//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TEXT_ENCODING_TEXT_GET_LOCALE_H
#define _LIBCPP___TEXT_ENCODING_TEXT_GET_LOCALE_H

#include <__config>
#include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD

string_view _LIBCPP_EXPORTED_FROM_ABI __get_locale_encoding(const char* __name);

_LIBCPP_END_NAMESPACE_STD
#endif 
