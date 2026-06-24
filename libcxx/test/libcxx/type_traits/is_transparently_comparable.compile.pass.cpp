//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that `__is_transparently_comparable_v` is true when we expect it to be

#include <functional>
#include <string>
#include <string_view>
#include <__type_traits/desugars_to.h>

// basic_string with char pointers/arrays (std::less)
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, const char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, char[5]>, "");

static_assert(std::__is_transparently_comparable_v<std::less<std::string>, const char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, char[5], std::string>, "");

// basic_string with char pointers/arrays (std::greater)
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, std::string, const char*>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, std::string, char*>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, std::string, char[5]>, "");

static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, const char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, char[5], std::string>, "");

// basic_string negative tests: wrong comparator
static_assert(
    !std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, std::string, char[5]>, "");
static_assert(
    !std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, char[5], std::string>, "");
static_assert(
    !std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, std::string, char const*>,
    "");
static_assert(
    !std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, char const*, std::string>,
    "");

// basic_string negative tests: mismatched character type
static_assert(!std::__is_transparently_comparable_v<std::less<std::string>, std::string, const wchar_t*>, "");
static_assert(!std::__is_transparently_comparable_v<std::less<std::string>, const wchar_t*, std::string>, "");

// basic_string_view with char pointers/arrays (std::less)
static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, std::string_view, const char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, std::string_view, char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, std::string_view, char[5]>, "");

static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, const char*, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, char*, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string_view>, char[5], std::string_view>, "");

// basic_string_view with char pointers/arrays (std::greater)
static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, std::string_view, const char*>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, std::string_view, char*>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, std::string_view, char[5]>, "");

static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, const char*, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, char*, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string_view>, char[5], std::string_view>, "");

// basic_string_view negative tests: wrong comparator
static_assert(!std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string_view> >,
                                                    std::string_view,
                                                    const char*>,
              "");
static_assert(!std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string_view> >,
                                                    const char*,
                                                    std::string_view>,
              "");
static_assert(!std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string_view> >,
                                                    std::string_view,
                                                    char[5]>,
              "");
static_assert(!std::__is_transparently_comparable_v<std::less<std::reference_wrapper<std::string_view> >,
                                                    char[5],
                                                    std::string_view>,
              "");

// basic_string_view negative tests: mismatched character type
static_assert(!std::__is_transparently_comparable_v<std::less<std::string_view>, std::string_view, const wchar_t*>, "");
static_assert(!std::__is_transparently_comparable_v<std::less<std::string_view>, const wchar_t*, std::string_view>, "");

// Cross-type: basic_string key with basic_string_view argument (std::less)
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string_view, std::string>, "");

// Cross-type: basic_string key with basic_string_view argument (std::greater)
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, std::string, std::string_view>, "");
static_assert(std::__is_transparently_comparable_v<std::greater<std::string>, std::string_view, std::string>, "");

// Cross-type negative: wrong comparator
static_assert(
    !std::
        __is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, std::string, std::string_view>,
    "");
static_assert(
    !std::
        __is_transparently_comparable_v<std::less<std::reference_wrapper<std::string> >, std::string_view, std::string>,
    "");
