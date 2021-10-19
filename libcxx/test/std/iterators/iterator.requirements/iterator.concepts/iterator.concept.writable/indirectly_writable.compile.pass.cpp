//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In>
// concept indirectly_writable;

#include <iterator>

#include <concepts>
#include <memory>
#include <optional>

#include "read_write.h"

template <class Out, class T>
constexpr bool check_indirectly_writable() {
  constexpr bool result = std::indirectly_writable<Out, T>;
  static_assert(std::indirectly_writable<Out const, T> == result);
  return result;
}

static_assert(check_indirectly_writable<value_type_indirection, int>());
static_assert(check_indirectly_writable<value_type_indirection, double>());
static_assert(!check_indirectly_writable<value_type_indirection, double*>());

static_assert(!check_indirectly_writable<read_only_indirection, int>());
static_assert(!check_indirectly_writable<proxy_indirection, int>());

static_assert(!check_indirectly_writable<int, int>());
static_assert(!check_indirectly_writable<missing_dereference, missing_dereference::value_type>());

static_assert(!check_indirectly_writable<void*, int>());
static_assert(!check_indirectly_writable<void const*, int>());
static_assert(!check_indirectly_writable<void volatile*, int>());
static_assert(!check_indirectly_writable<void const volatile*, int>());
static_assert(!check_indirectly_writable<void*, double>());
static_assert(check_indirectly_writable<void**, int*>());
static_assert(!check_indirectly_writable<void**, int>());

static_assert(check_indirectly_writable<int*, int>());
static_assert(!check_indirectly_writable<int const*, int>());
static_assert(check_indirectly_writable<int volatile*, int>());
static_assert(!check_indirectly_writable<int const volatile*, int>());
static_assert(check_indirectly_writable<int*, double>());
static_assert(check_indirectly_writable<int**, int*>());
static_assert(!check_indirectly_writable<int**, int>());
