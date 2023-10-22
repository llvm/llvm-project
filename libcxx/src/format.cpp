//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <format>

_LIBCPP_BEGIN_NAMESPACE_STD

template back_insert_iterator<string>
    vformat_to<back_insert_iterator<string>>(back_insert_iterator<string>, string_view, format_args);

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template back_insert_iterator<wstring>
    vformat_to<back_insert_iterator<wstring>>(back_insert_iterator<wstring>, wstring_view, wformat_args);
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS

format_error::~format_error() noexcept = default;

_LIBCPP_END_NAMESPACE_STD
