//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

#include <__type_traits/is_equality_comparable.h>

enum Enum : int {};
enum class EnumClass : int {};

static_assert(std::__is_trivially_equality_comparable<int, int>::value, "");
static_assert(std::__is_trivially_equality_comparable<const int, int>::value, "");
static_assert(std::__is_trivially_equality_comparable<int, const int>::value, "");

static_assert(std::__is_trivially_equality_comparable<unsigned int, unsigned int>::value, "");
static_assert(std::__is_trivially_equality_comparable<const unsigned int, unsigned int>::value, "");
static_assert(!std::__is_trivially_equality_comparable<unsigned int, int>::value, "");

static_assert(!std::__is_trivially_equality_comparable<long, int>::value, "");
static_assert(!std::__is_trivially_equality_comparable<int, long>::value, "");

static_assert(std::__is_trivially_equality_comparable<int*, int*>::value, "");
static_assert(std::__is_trivially_equality_comparable<int*, void*>::value, "");
static_assert(!std::__is_trivially_equality_comparable<int*, long*>::value, "");

static_assert(!std::__is_trivially_equality_comparable<Enum, int>::value, "");
static_assert(!std::__is_trivially_equality_comparable<EnumClass, int>::value, "");

static_assert(!std::__is_trivially_equality_comparable<float, int>::value, "");
static_assert(!std::__is_trivially_equality_comparable<double, long long>::value, "");

static_assert(!std::__is_trivially_equality_comparable<float, int>::value, "");

static_assert(!std::__is_trivially_equality_comparable<float, float>::value, "");
static_assert(!std::__is_trivially_equality_comparable<double, double>::value, "");
static_assert(!std::__is_trivially_equality_comparable<long double, long double>::value, "");

struct S {
  char c;
};

struct S2 {
  char c;
};

struct VirtualBase : virtual S {};
struct NonVirtualBase : S, S2 {};

static_assert(!std::__is_trivially_equality_comparable<S*, VirtualBase*>::value, "");
static_assert(!std::__is_trivially_equality_comparable<S2*, VirtualBase*>::value, "");

// This is trivially_equality_comparable, but we can't detect it currently
static_assert(!std::__is_trivially_equality_comparable<S*, NonVirtualBase*>::value, "");
