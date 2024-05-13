//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template <class _Tp, template <class...> class _Template>
// inline constexpr bool __is_specialization_v = true if and only if _Tp is a specialization of _Template
//
// Note instantiation for certain type combinations are ill-formed. These are
// tested in is_specialization.verify.cpp.

#include <type_traits>

#include <array>
#include <concepts>
#include <string_view>
#include <tuple>
#include <utility>

#include "test_macros.h"

// Simple types
static_assert(std::__is_specialization_v<std::pair<int, int>, std::pair>);
static_assert(!std::__is_specialization_v<std::pair<int, int>, std::tuple>);
static_assert(!std::__is_specialization_v<std::pair<int, int>, std::basic_string_view>);

static_assert(std::__is_specialization_v<std::tuple<int>, std::tuple>);
static_assert(std::__is_specialization_v<std::tuple<int, float>, std::tuple>);
static_assert(std::__is_specialization_v<std::tuple<int, float, void*>, std::tuple>);

static_assert(std::__is_specialization_v<std::string_view, std::basic_string_view>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::__is_specialization_v<std::wstring_view, std::basic_string_view>);
#endif

// Nested types
static_assert(std::__is_specialization_v<std::pair<std::tuple<int>, int>, std::pair>);
static_assert(!std::__is_specialization_v<std::pair<std::tuple<int>, int>, std::tuple>);

// cvref _Tp is not a specialization.
static_assert(!std::__is_specialization_v<const std::pair<int, int>, std::pair>);
static_assert(!std::__is_specialization_v<volatile std::pair<int, int>, std::pair>);
static_assert(!std::__is_specialization_v<const volatile std::pair<int, int>, std::pair>);

static_assert(!std::__is_specialization_v<std::pair<int, int>&, std::pair>);
static_assert(!std::__is_specialization_v<const std::pair<int, int>&, std::pair>);
static_assert(!std::__is_specialization_v<volatile std::pair<int, int>&, std::pair>);
static_assert(!std::__is_specialization_v<const volatile std::pair<int, int>&, std::pair>);

static_assert(!std::__is_specialization_v<std::pair<int, int>&&, std::pair>);
static_assert(!std::__is_specialization_v<const std::pair<int, int>&&, std::pair>);
static_assert(!std::__is_specialization_v<volatile std::pair<int, int>&&, std::pair>);
static_assert(!std::__is_specialization_v<const volatile std::pair<int, int>&&, std::pair>);
