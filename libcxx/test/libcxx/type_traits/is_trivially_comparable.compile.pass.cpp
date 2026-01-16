//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/conditional.h>
#include <__type_traits/is_equality_comparable.h>
#include <__type_traits/is_signed.h>
#include <cstdint>

enum Enum : int {};
enum class EnumClass : int {};

static_assert(std::__is_trivially_equality_comparable_v<int, int>, "");
static_assert(std::__is_trivially_equality_comparable_v<const int, int>, "");
static_assert(std::__is_trivially_equality_comparable_v<int, const int>, "");

static_assert(std::__is_trivially_equality_comparable_v<unsigned int, unsigned int>, "");
static_assert(std::__is_trivially_equality_comparable_v<const unsigned int, unsigned int>, "");
static_assert(!std::__is_trivially_equality_comparable_v<unsigned int, int>, "");

static_assert(!std::__is_trivially_equality_comparable_v<std::int32_t, std::int64_t>, "");
static_assert(!std::__is_trivially_equality_comparable_v<std::int64_t, std::int32_t>, "");

static_assert(std::__is_trivially_equality_comparable_v<int*, int*>, "");
static_assert(std::__is_trivially_equality_comparable_v<int*, void*>, "");
static_assert(!std::__is_trivially_equality_comparable_v<int*, long*>, "");

static_assert(!std::__is_trivially_equality_comparable_v<Enum, int>, "");
static_assert(!std::__is_trivially_equality_comparable_v<EnumClass, int>, "");

static_assert(!std::__is_trivially_equality_comparable_v<float, int>, "");
static_assert(!std::__is_trivially_equality_comparable_v<double, long long>, "");

static_assert(!std::__is_trivially_equality_comparable_v<float, int>, "");

static_assert(!std::__is_trivially_equality_comparable_v<float, float>, "");
static_assert(!std::__is_trivially_equality_comparable_v<double, double>, "");
static_assert(!std::__is_trivially_equality_comparable_v<long double, long double>, "");

static_assert(std::__is_trivially_equality_comparable_v<
                  char,
                  typename std::conditional<std::is_signed<char>::value, signed char, unsigned char>::type>,
              "");
static_assert(std::__is_trivially_equality_comparable_v<char16_t, std::uint_least16_t>, "");

struct S {
  char c;
};

struct S2 {
  char c;
};

struct VirtualBase : virtual S {};
struct NonVirtualBase : S, S2 {};

static_assert(!std::__is_trivially_equality_comparable_v<S*, VirtualBase*>, "");
static_assert(!std::__is_trivially_equality_comparable_v<S2*, VirtualBase*>, "");

// This is trivially_equality_comparable, but we can't detect it currently
static_assert(!std::__is_trivially_equality_comparable_v<S*, NonVirtualBase*>, "");
