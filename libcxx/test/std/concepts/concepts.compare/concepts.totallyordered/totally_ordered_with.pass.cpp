//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept totally_ordered_with;

#include <concepts>

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../types.h"
#include "test_macros.h"

template <class T, class U>
constexpr bool check_totally_ordered_with() noexcept {
  constexpr bool result = std::totally_ordered_with<T, U>;
  static_assert(std::totally_ordered_with<U, T> == result);
  static_assert(std::totally_ordered_with<T, U const> == result);
  static_assert(std::totally_ordered_with<T const, U const> == result);
  static_assert(std::totally_ordered_with<T, U const&> == result);
  static_assert(std::totally_ordered_with<T const, U const&> == result);
  static_assert(std::totally_ordered_with<T&, U const> == result);
  static_assert(std::totally_ordered_with<T const&, U const> == result);
  static_assert(std::totally_ordered_with<T&, U const&> == result);
  static_assert(std::totally_ordered_with<T const&, U const&> == result);
  static_assert(std::totally_ordered_with<T, U const&&> == result);
  static_assert(std::totally_ordered_with<T const, U const&&> == result);
  static_assert(std::totally_ordered_with<T&, U const&&> == result);
  static_assert(std::totally_ordered_with<T const&, U const&&> == result);
  static_assert(std::totally_ordered_with<T&&, U const> == result);
  static_assert(std::totally_ordered_with<T const&&, U const> == result);
  static_assert(std::totally_ordered_with<T&&, U const&> == result);
  static_assert(std::totally_ordered_with<T const&&, U const&> == result);
  static_assert(std::totally_ordered_with<T&&, U const&&> == result);
  static_assert(std::totally_ordered_with<T const&&, U const&&> == result);
  return result;
}

namespace fundamentals {
static_assert(check_totally_ordered_with<int, int>());
static_assert(check_totally_ordered_with<int, bool>());
static_assert(check_totally_ordered_with<int, char>());
static_assert(check_totally_ordered_with<int, wchar_t>());
static_assert(check_totally_ordered_with<int, double>());
static_assert(!check_totally_ordered_with<int, int*>());
static_assert(!check_totally_ordered_with<int, int[5]>());
static_assert(!check_totally_ordered_with<int, int (*)()>());
static_assert(!check_totally_ordered_with<int, int (&)()>());

struct S {};
static_assert(!check_totally_ordered_with<int, int S::*>());
static_assert(!check_totally_ordered_with<int, int (S::*)()>());
static_assert(!check_totally_ordered_with<int, int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() const>());
static_assert(!check_totally_ordered_with<int, int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile>());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile>());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int, int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile&>());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int, int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int,
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int, int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int,
              int (S::*)() const volatile&& noexcept > ());

static_assert(check_totally_ordered_with<int*, int*>());
static_assert(check_totally_ordered_with<int*, int[5]>());
static_assert(!check_totally_ordered_with<int*, int (*)()>());
static_assert(!check_totally_ordered_with<int*, int (&)()>());
static_assert(!check_totally_ordered_with<int*, int (S::*)()>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile>());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile>());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile&>());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile &
                                                    noexcept>());
static_assert(!check_totally_ordered_with<int*, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int*,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int*, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int*,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int*,
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int*, int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int*,
              int (S::*)() const volatile&& noexcept > ());

static_assert(check_totally_ordered_with<int[5], int[5]>());
static_assert(!check_totally_ordered_with<int[5], int (*)()>());
static_assert(!check_totally_ordered_with<int[5], int (&)()>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)()>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int[5], int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() &>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile&>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile &
                                                      noexcept>());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int[5],
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int[5],
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int[5],
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int[5], int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int[5],
              int (S::*)() const volatile&& noexcept > ());

static_assert(check_totally_ordered_with<int (*)(), int (*)()>());
static_assert(check_totally_ordered_with<int (*)(), int (&)()>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)()>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (*)(), int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile&>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (*)(), int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (*)(),
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (*)(),
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (*)(),
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (*)(), int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (*)(),
              int (S::*)() const volatile&& noexcept > ());

static_assert(check_totally_ordered_with<int (&)(), int (&)()>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)()>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (&)(), int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile&>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (&)(), int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (&)(),
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (&)(),
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (&)(),
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (&)(), int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (&)(),
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)()>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<int (S::*)(),
                                          int (S::*)() volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (S::*)(), int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const&>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)(),
                                          int (S::*)() volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)(), int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)(),
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)(),
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)(),
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)(),
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (S::*)() noexcept, int (S::*)() const volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)() noexcept, int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() noexcept,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() const>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() const noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (S::*)() const, int (S::*)() const volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() const & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)() const, int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const,
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const volatile>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const noexcept,
                                int (S::*)() const volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              int (S::*)() volatile, int (S::*)() const volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)() volatile, int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const volatile>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile noexcept,
                                int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const volatile>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile,
                                int (S::*)() const volatile noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile,
              int (S::*)() const volatile&& noexcept > ());

static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() &>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<int (S::*)() &, int (S::*)() & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() &, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() &,
                                          int (S::*)() const & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() &, int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() &,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() &,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)() &, int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() &,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() &, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &,
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() &, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() &,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() & noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() & noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() & noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() & noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() & noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(
    !check_totally_ordered_with<int (S::*)() const&, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<int (S::*)() const&,
                                          int (S::*)() const & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const&, int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const&,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const&,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              int (S::*)() const&, int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const&, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() const&, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const&,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() const & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const & noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const& noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const& noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const& noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const& noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile&,
                                int (S::*)() const volatile & noexcept>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile& noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile& noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile& noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile& noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() const volatile&,
                                          int (S::*)() const volatile&>());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile&,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&,
                                          int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile&,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&,
                                          int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile&,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile & noexcept,
                                int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<
              int (S::*)() const volatile & noexcept, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile& noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with<
              int (S::*)() const volatile & noexcept, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile& noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile & noexcept,
                                int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile& noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() const volatile & noexcept,
                                int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile& noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() &&, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < int (S::*)() &&,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() &&, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &&,
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<int (S::*)() &&, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() &&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() &&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() && noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() const&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() volatile&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() const volatile&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() && noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(
    !check_totally_ordered_with<int (S::*)() const&&, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&&,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const&&,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() const&&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const&&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with < int (S::*)() const&& noexcept,
              int (S::*)() const&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() const&& noexcept,
              int (S::*)() volatile&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() const&& noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() const&& noexcept,
              int (S::*)() const volatile&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() const&& noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() volatile&&,
                                          int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&&,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<int (S::*)() volatile&&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&&,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with < int (S::*)() volatile&& noexcept,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&& noexcept,
              int (S::*)() const volatile&& > ());
static_assert(!check_totally_ordered_with < int (S::*)() volatile&& noexcept,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!check_totally_ordered_with<int (S::*)() const volatile&&,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < int (S::*)() const volatile&&,
              int (S::*)() const volatile&& noexcept > ());
static_assert(!check_totally_ordered_with < int (S::*)()
                                                const volatile&& noexcept,
              int (S::*)() const volatile&& noexcept > ());

#if !defined(TEST_COMPILER_GCC)
static_assert(!check_totally_ordered_with<std::nullptr_t, int>());

static_assert(!check_totally_ordered_with<std::nullptr_t, int*>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int[]>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int[5]>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int (*)()>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int (&)()>());
#endif

static_assert(!check_totally_ordered_with<std::nullptr_t, int (S::*)()>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() noexcept>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() const>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() const noexcept>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() volatile>());
static_assert(!check_totally_ordered_with<std::nullptr_t,
                                          int (S::*)() volatile noexcept>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() const volatile>());
static_assert(!check_totally_ordered_with<
              std::nullptr_t, int (S::*)() const volatile noexcept>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int (S::*)() &>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() & noexcept>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() const&>());
static_assert(!check_totally_ordered_with<std::nullptr_t,
                                          int (S::*)() const & noexcept>());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() volatile&>());
static_assert(!check_totally_ordered_with<std::nullptr_t,
                                          int (S::*)() volatile & noexcept>());
static_assert(!check_totally_ordered_with<std::nullptr_t,
                                          int (S::*)() const volatile&>());
static_assert(!check_totally_ordered_with<
              std::nullptr_t, int (S::*)() const volatile & noexcept>());
static_assert(!check_totally_ordered_with<std::nullptr_t, int (S::*)() &&>());
static_assert(!check_totally_ordered_with < std::nullptr_t,
              int (S::*)() && noexcept > ());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() const&&>());
static_assert(!check_totally_ordered_with < std::nullptr_t,
              int (S::*)() const&& noexcept > ());
static_assert(
    !check_totally_ordered_with<std::nullptr_t, int (S::*)() volatile&&>());
static_assert(!check_totally_ordered_with < std::nullptr_t,
              int (S::*)() volatile&& noexcept > ());
static_assert(!check_totally_ordered_with<std::nullptr_t,
                                          int (S::*)() const volatile&&>());
static_assert(!check_totally_ordered_with < std::nullptr_t,
              int (S::*)() const volatile&& noexcept > ());

static_assert(!std::equality_comparable_with<void, int>);
static_assert(!std::equality_comparable_with<void, int*>);
static_assert(!std::equality_comparable_with<void, std::nullptr_t>);
static_assert(!std::equality_comparable_with<void, int[5]>);
static_assert(!std::equality_comparable_with<void, int (*)()>);
static_assert(!std::equality_comparable_with<void, int (&)()>);
static_assert(!std::equality_comparable_with<void, int S::*>);
static_assert(!std::equality_comparable_with<void, int (S::*)()>);
static_assert(!std::equality_comparable_with<void, int (S::*)() noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() const>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() volatile>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() volatile noexcept>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const volatile>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const volatile noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() &>);
static_assert(!std::equality_comparable_with<void, int (S::*)() & noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() const&>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const & noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() volatile&>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() volatile & noexcept>);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const volatile&>);
static_assert(!std::equality_comparable_with<void, int (S::*)() const volatile &
                                                       noexcept>);
static_assert(!std::equality_comparable_with<void, int (S::*)() &&>);
static_assert(!std::equality_comparable_with < void,
              int (S::*)() && noexcept >);
static_assert(!std::equality_comparable_with<void, int (S::*)() const&&>);
static_assert(!std::equality_comparable_with < void,
              int (S::*)() const&& noexcept >);
static_assert(!std::equality_comparable_with<void, int (S::*)() volatile&&>);
static_assert(!std::equality_comparable_with < void,
              int (S::*)() volatile&& noexcept >);
static_assert(
    !std::equality_comparable_with<void, int (S::*)() const volatile&&>);
static_assert(!std::equality_comparable_with < void,
              int (S::*)() const volatile&& noexcept >);
} // namespace fundamentals

namespace standard_types {
static_assert(
    check_totally_ordered_with<std::array<int, 10>, std::array<int, 10> >());
static_assert(!check_totally_ordered_with<std::array<int, 10>,
                                          std::array<double, 10> >());
static_assert(check_totally_ordered_with<std::deque<int>, std::deque<int> >());
static_assert(
    !check_totally_ordered_with<std::deque<int>, std::vector<int> >());
static_assert(check_totally_ordered_with<std::forward_list<int>,
                                         std::forward_list<int> >());
static_assert(
    !check_totally_ordered_with<std::forward_list<int>, std::vector<int> >());
static_assert(check_totally_ordered_with<std::list<int>, std::list<int> >());
static_assert(!check_totally_ordered_with<std::list<int>, std::vector<int> >());

static_assert(
    check_totally_ordered_with<std::map<int, void*>, std::map<int, void*> >());
static_assert(
    !check_totally_ordered_with<std::map<int, void*>, std::vector<int> >());
static_assert(check_totally_ordered_with<std::optional<std::vector<int> >,
                                         std::optional<std::vector<int> > >());
static_assert(check_totally_ordered_with<std::optional<std::vector<int> >,
                                         std::vector<int> >());
static_assert(
    check_totally_ordered_with<std::vector<int>, std::vector<int> >());
static_assert(!check_totally_ordered_with<std::vector<int>, int>());

struct A {};
// FIXME(cjdb): uncomment when operator<=> is implemented for each of these types.
// static_assert(!check_totally_ordered_with<std::optional<std::vector<A> >,
//                                          std::optional<std::vector<A> > >());
// static_assert(!check_totally_ordered_with<std::optional<std::vector<A> >,
//                                          std::vector<A> >());
struct B {};
static_assert(!check_totally_ordered_with<std::vector<A>, std::vector<B> >());
static_assert(
    !check_totally_ordered_with<std::optional<A>, std::optional<B> >());
} // namespace standard_types

namespace types_fit_for_purpose {
static_assert(!check_totally_ordered_with<cxx20_member_eq, cxx20_member_eq>());
static_assert(!check_totally_ordered_with<cxx20_friend_eq, cxx20_friend_eq>());
static_assert(!check_totally_ordered_with<cxx20_member_eq, cxx20_friend_eq>());

static_assert(check_totally_ordered_with<member_three_way_comparable,
                                         member_three_way_comparable>());
static_assert(check_totally_ordered_with<friend_three_way_comparable,
                                         friend_three_way_comparable>());
static_assert(!check_totally_ordered_with<member_three_way_comparable,
                                          friend_three_way_comparable>());

static_assert(
    check_totally_ordered_with<explicit_operators, explicit_operators>());
static_assert(!check_totally_ordered_with<equality_comparable_with_ec1,
                                          equality_comparable_with_ec1>());
static_assert(check_totally_ordered_with<different_return_types,
                                         different_return_types>());
static_assert(!check_totally_ordered_with<explicit_operators,
                                          equality_comparable_with_ec1>());
static_assert(
    check_totally_ordered_with<explicit_operators, different_return_types>());

static_assert(!check_totally_ordered_with<one_way_eq, one_way_eq>());
static_assert(
    std::common_reference_with<one_way_eq const&, explicit_operators const&> &&
    !check_totally_ordered_with<one_way_eq, explicit_operators>());

static_assert(!check_totally_ordered_with<one_way_ne, one_way_ne>());
static_assert(
    std::common_reference_with<one_way_ne const&, explicit_operators const&> &&
    !check_totally_ordered_with<one_way_ne, explicit_operators>());

static_assert(
    check_totally_ordered_with<totally_ordered_with_others,
                               partial_ordering_totally_ordered_with>());
static_assert(check_totally_ordered_with<totally_ordered_with_others,
                                         weak_ordering_totally_ordered_with>());
static_assert(
    check_totally_ordered_with<totally_ordered_with_others,
                               strong_ordering_totally_ordered_with>());

static_assert(!check_totally_ordered_with<totally_ordered_with_others,
                                          eq_returns_explicit_bool>());
static_assert(!check_totally_ordered_with<totally_ordered_with_others,
                                          ne_returns_explicit_bool>());
static_assert(std::equality_comparable_with<totally_ordered_with_others,
                                            lt_returns_explicit_bool> &&
              !check_totally_ordered_with<totally_ordered_with_others,
                                          lt_returns_explicit_bool>());
static_assert(std::equality_comparable_with<totally_ordered_with_others,
                                            gt_returns_explicit_bool> &&
              !check_totally_ordered_with<totally_ordered_with_others,
                                          gt_returns_explicit_bool>());
static_assert(std::equality_comparable_with<totally_ordered_with_others,
                                            le_returns_explicit_bool> &&
              !check_totally_ordered_with<totally_ordered_with_others,
                                          le_returns_explicit_bool>());
static_assert(std::equality_comparable_with<totally_ordered_with_others,
                                            ge_returns_explicit_bool> &&
              !check_totally_ordered_with<totally_ordered_with_others,
                                          ge_returns_explicit_bool>());
static_assert(check_totally_ordered_with<totally_ordered_with_others,
                                         returns_true_type>());
static_assert(
    check_totally_ordered_with<totally_ordered_with_others, returns_int_ptr>());

static_assert(
    std::totally_ordered<no_lt_not_totally_ordered_with>&&
        std::equality_comparable_with<totally_ordered_with_others,
                                      no_lt_not_totally_ordered_with> &&
    !check_totally_ordered_with<totally_ordered_with_others,
                                no_lt_not_totally_ordered_with>());
static_assert(
    std::totally_ordered<no_gt_not_totally_ordered_with>&&
        std::equality_comparable_with<totally_ordered_with_others,
                                      no_gt_not_totally_ordered_with> &&
    !check_totally_ordered_with<totally_ordered_with_others,
                                no_gt_not_totally_ordered_with>());
static_assert(
    std::totally_ordered<no_le_not_totally_ordered_with>&&
        std::equality_comparable_with<totally_ordered_with_others,
                                      no_le_not_totally_ordered_with> &&
    !check_totally_ordered_with<totally_ordered_with_others,
                                no_le_not_totally_ordered_with>());
static_assert(
    std::totally_ordered<no_ge_not_totally_ordered_with>&&
        std::equality_comparable_with<totally_ordered_with_others,
                                      no_ge_not_totally_ordered_with> &&
    !check_totally_ordered_with<totally_ordered_with_others,
                                no_ge_not_totally_ordered_with>());
} // namespace types_fit_for_purpose

int main(int, char**) { return 0; }
