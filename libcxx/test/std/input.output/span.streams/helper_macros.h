//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUTOUTPUT_SPANSTREAMS_MACROS_H
#define TEST_STD_INPUTOUTPUT_SPANSTREAMS_MACROS_H

#include <type_traits>

#include "make_string.h"

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#  define CH(C)                                                                                                        \
    (std::is_same_v<CharT, wchar_t>    ? L##C                                                                          \
     : std::is_same_v<CharT, char8_t>  ? u8##C                                                                         \
     : std::is_same_v<CharT, char16_t> ? u##C                                                                          \
     : std::is_same_v<CharT, char32_t> ? U##C                                                                          \
                                       : C)
#else
#  define CH(C)                                                                                                        \
    (std::is_same_v<CharT, char8_t>    ? u8##C                                                                         \
     : std::is_same_v<CharT, char16_t> ? u##C                                                                          \
     : std::is_same_v<CharT, char32_t> ? U##C                                                                          \
                                       : C)
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S, a) std::basic_string<CharT, TraitsT, AllocT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S), a)
#define SV(S) std::basic_string_view<CharT, TraitsT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S))

#endif // TEST_STD_INPUTOUTPUT_SPANSTREAMS_MACROS_H
