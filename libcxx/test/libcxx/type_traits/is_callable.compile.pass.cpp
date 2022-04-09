//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

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

static_assert(std::__is_callable<Functor>::value, "");
static_assert(std::__is_callable<decltype(func)>::value, "");
static_assert(!std::__is_callable<NotFunctor>::value, "");
static_assert(!std::__is_callable<NotFunctor,
                                  decltype(&NotFunctor::compare)>::value, "");
static_assert(std::__is_callable<ArgumentFunctor, int, int>::value, "");
static_assert(!std::__is_callable<ArgumentFunctor, int>::value, "");
