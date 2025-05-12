//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/is_callable.h>

struct Functor {
  void operator()();
};

int func();

struct NotFunctor {
  bool compare();
};

struct ArgumentFunctor {
  bool operator()(int, int);
};

static_assert(std::__is_callable_v<Functor>, "");
static_assert(std::__is_callable_v<decltype(func)>, "");
static_assert(!std::__is_callable_v<NotFunctor>, "");
static_assert(!std::__is_callable_v<NotFunctor, decltype(&NotFunctor::compare)>, "");
static_assert(std::__is_callable_v<ArgumentFunctor, int, int>, "");
static_assert(!std::__is_callable_v<ArgumentFunctor, int>, "");
