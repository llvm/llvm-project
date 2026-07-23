//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

#include <tuple>
#include <type_traits>

#include "test_macros.h"

struct S {};

struct Final final {};

struct NonEmpty {
  int i;
  char c;
};

struct NonEmptyFinal final {
  int i;
  char c;
};

struct TEST_ALIGNAS(16) Overaligned {};
struct TEST_ALIGNAS(16) OveralignedFinal final {};

static_assert(std::is_empty<std::tuple<>>::value, "");
static_assert(!std::is_empty<std::tuple<S>>::value, "");
static_assert(!std::is_empty<std::tuple<S&>>::value, "");
static_assert(!std::is_empty<std::tuple<S&&>>::value, "");
static_assert(!std::is_empty<std::tuple<Final>>::value, "");
static_assert(!std::is_empty<std::tuple<NonEmpty>>::value, "");
static_assert(!std::is_empty<std::tuple<NonEmptyFinal>>::value, "");
static_assert(!std::is_empty<std::tuple<Overaligned>>::value, "");
static_assert(!std::is_empty<std::tuple<OveralignedFinal>>::value, "");

static_assert(sizeof(std::tuple<S>) == 1, "");
static_assert(sizeof(std::tuple<S&>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<S&&>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<Final>) == 1, "");
static_assert(sizeof(std::tuple<NonEmpty>) == 8, "");
static_assert(sizeof(std::tuple<NonEmptyFinal>) == 8, "");
static_assert(sizeof(std::tuple<Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal>) == 16, "");

static_assert(sizeof(std::tuple<S, S>) == 2, "");
static_assert(sizeof(std::tuple<S&, S>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<S&&, S>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<Final, S>) == 1, "");
static_assert(sizeof(std::tuple<NonEmpty, S>) == 8, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, S>) == 8, "");
static_assert(sizeof(std::tuple<Overaligned, S>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, S>) == 16, "");

static_assert(sizeof(std::tuple<S, S&>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<S&, S&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<S&&, S&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<Final, S&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<NonEmpty, S&>) == 8 + sizeof(void*), "");
static_assert(sizeof(std::tuple<NonEmptyFinal, S&>) == 8 + sizeof(void*), "");
static_assert(sizeof(std::tuple<Overaligned, S&>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, S&>) == 32, "");

static_assert(sizeof(std::tuple<S, S&&>) == sizeof(void*), "");
static_assert(sizeof(std::tuple<S&, S&&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<S&&, S&&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<Final, S&&>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<NonEmpty, S&&>) == 8 + sizeof(void*), "");
static_assert(sizeof(std::tuple<NonEmptyFinal, S&&>) == 8 + sizeof(void*), "");
static_assert(sizeof(std::tuple<Overaligned, S&&>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, S&&>) == 32, "");

static_assert(sizeof(std::tuple<S, Final>) == 1, "");
static_assert(sizeof(std::tuple<S&, Final>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<S&&, Final>) == 2 * sizeof(void*), "");
static_assert(sizeof(std::tuple<Final, Final>) == 2, "");
static_assert(sizeof(std::tuple<NonEmpty, Final>) == 12, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, Final>) == 12, "");
static_assert(sizeof(std::tuple<Overaligned, Final>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, Final>) == 32, "");

static_assert(sizeof(std::tuple<S, NonEmpty>) == 8, "");
static_assert(sizeof(std::tuple<S&, NonEmpty>) == sizeof(void*) + 8, "");
static_assert(sizeof(std::tuple<S&&, NonEmpty>) == sizeof(void*) + 8, "");
static_assert(sizeof(std::tuple<Final, NonEmpty>) == 12, "");
static_assert(sizeof(std::tuple<NonEmpty, NonEmpty>) == 16, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, NonEmpty>) == 16, "");
static_assert(sizeof(std::tuple<Overaligned, NonEmpty>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, NonEmpty>) == 32, "");

static_assert(sizeof(std::tuple<S, NonEmptyFinal>) == 8, "");
static_assert(sizeof(std::tuple<S&, NonEmptyFinal>) == sizeof(void*) + 8, "");
static_assert(sizeof(std::tuple<S&&, NonEmptyFinal>) == sizeof(void*) + 8, "");
static_assert(sizeof(std::tuple<Final, NonEmptyFinal>) == 12, "");
static_assert(sizeof(std::tuple<NonEmpty, NonEmptyFinal>) == 16, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, NonEmptyFinal>) == 16, "");
static_assert(sizeof(std::tuple<Overaligned, NonEmptyFinal>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, NonEmptyFinal>) == 32, "");

static_assert(sizeof(std::tuple<S, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<S&, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<S&&, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<Final, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<NonEmpty, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, Overaligned>) == 16, "");
static_assert(sizeof(std::tuple<Overaligned, Overaligned>) == 32, "");
static_assert(sizeof(std::tuple<OveralignedFinal, Overaligned>) == 16, "");

static_assert(sizeof(std::tuple<S, OveralignedFinal>) == 16, "");
static_assert(sizeof(std::tuple<S&, OveralignedFinal>) == 32, "");
static_assert(sizeof(std::tuple<S&&, OveralignedFinal>) == 32, "");
static_assert(sizeof(std::tuple<Final, OveralignedFinal>) == 32, "");
static_assert(sizeof(std::tuple<NonEmpty, OveralignedFinal>) == 32, "");
static_assert(sizeof(std::tuple<NonEmptyFinal, OveralignedFinal>) == 32, "");
static_assert(sizeof(std::tuple<Overaligned, OveralignedFinal>) == 16, "");
static_assert(sizeof(std::tuple<OveralignedFinal, OveralignedFinal>) == 32, "");
