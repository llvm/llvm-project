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
#include <__type_traits/desugars_to.h>

static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, const char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, char*>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, std::string, char[5]>, "");

static_assert(std::__is_transparently_comparable_v<std::less<std::string>, const char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, char*, std::string>, "");
static_assert(std::__is_transparently_comparable_v<std::less<std::string>, char[5], std::string>, "");

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
