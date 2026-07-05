//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This test makes sure that we can copy/move a std::tuple containing a type
// that checks for copy constructibility itself, like std::any.
//
// Problem showcased in https://reviews.llvm.org/D96523#2730953.

#include <any>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class ...Pred>
struct And : std::true_type { };

template <class P1, class ...Pn>
struct And<P1, Pn...>
  : std::conditional<P1::value, And<Pn...>, std::false_type>::type
{ };

struct any {
  any();
  any(any const&) = default;

  template <class ValueType,
            class Decayed = typename std::decay<ValueType>::type,
            class = typename std::enable_if<
              !std::is_same<Decayed, any>::value &&
              std::is_copy_constructible<Decayed>::value
            >::type>
  any(ValueType&&);
};

struct A {
  A();
  A(any);
};

#if TEST_STD_VER > 14
struct B {
  B();
  B(std::any);
};
#endif

void f() {
  {
    std::tuple<A, int> x;
    std::tuple<A, int> y = x; (void)y;
  }
  {
    std::tuple<A, int> x;
    std::tuple<A, int> y = std::move(x); (void)y;
  }

#if TEST_STD_VER > 14
  {
    std::tuple<B, int> x;
    std::tuple<B, int> y = x; (void)y;
  }
  {
    std::tuple<B, int> x;
    std::tuple<B, int> y = std::move(x); (void)y;
  }
#endif
}
