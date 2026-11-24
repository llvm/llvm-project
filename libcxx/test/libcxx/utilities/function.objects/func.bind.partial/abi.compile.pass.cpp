//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <functional>

template <class... Args>
using back_t = decltype(std::bind_back(std::declval<Args>()...));

template <class... Args>
using front_t = decltype(std::bind_back(std::declval<Args>()...));

struct Func {
  template <class... Args>
  void operator()(Args...);
};

struct Empty {};
struct Empty2 {};
struct EmptyFinal final {};

static_assert(sizeof(back_t<Func, Empty>) == 1);
static_assert(sizeof(back_t<Func, Empty, EmptyFinal>) == 1);
static_assert(sizeof(back_t<Func, Empty, EmptyFinal, Empty>) == 2);
static_assert(sizeof(back_t<Func, Empty, Empty>) == 2);
static_assert(sizeof(back_t<Func, Empty, Empty2>) == 1);

static_assert(sizeof(front_t<Func, Empty>) == 1);
static_assert(sizeof(front_t<Func, Empty, EmptyFinal>) == 1);
static_assert(sizeof(front_t<Func, Empty, EmptyFinal, Empty>) == 2);
static_assert(sizeof(front_t<Func, Empty, Empty>) == 2);
static_assert(sizeof(front_t<Func, Empty, Empty2>) == 1);

struct Big {
  char arr[10000];
};

static_assert(sizeof(back_t<Func, Big>) == 10000);
static_assert(sizeof(back_t<Func, Big, Empty>) == 10000);
static_assert(sizeof(back_t<Func, Big, Empty, Empty>) == 10001);

static_assert(sizeof(front_t<Func, Big>) == 10000);
static_assert(sizeof(front_t<Func, Big, Empty>) == 10000);
static_assert(sizeof(front_t<Func, Big, Empty, Empty>) == 10001);
