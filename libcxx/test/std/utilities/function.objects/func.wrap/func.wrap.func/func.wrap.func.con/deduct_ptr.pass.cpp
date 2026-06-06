//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class R, class ...Args>
// function(R(*)(Args...)) -> function<R(Args...)>;

// UNSUPPORTED: c++03, c++11, c++14

#include <functional>
#include <type_traits>

#include "test_macros.h"

struct R { };
struct A1 { };
struct A2 { };
struct A3 { };

R f0() { return {}; }
R f1(A1) { return {}; }
R f2(A1, A2) { return {}; }
R f3(A1, A2, A3) { return {}; }
R f4(A1 = {}) { return {}; }

int main(int, char**) {
  {
    // implicit
    std::function a = f0;
    ASSERT_SAME_TYPE(decltype(a), std::function<R()>);

    std::function b = &f0;
    ASSERT_SAME_TYPE(decltype(b), std::function<R()>);

    // explicit
    std::function c{f0};
    ASSERT_SAME_TYPE(decltype(c), std::function<R()>);

    std::function d{&f0};
    ASSERT_SAME_TYPE(decltype(d), std::function<R()>);
  }
  {
    // implicit
    std::function a = f1;
    ASSERT_SAME_TYPE(decltype(a), std::function<R(A1)>);

    std::function b = &f1;
    ASSERT_SAME_TYPE(decltype(b), std::function<R(A1)>);

    // explicit
    std::function c{f1};
    ASSERT_SAME_TYPE(decltype(c), std::function<R(A1)>);

    std::function d{&f1};
    ASSERT_SAME_TYPE(decltype(d), std::function<R(A1)>);
  }
  {
    // implicit
    std::function a = f2;
    ASSERT_SAME_TYPE(decltype(a), std::function<R(A1, A2)>);

    std::function b = &f2;
    ASSERT_SAME_TYPE(decltype(b), std::function<R(A1, A2)>);

    // explicit
    std::function c{f2};
    ASSERT_SAME_TYPE(decltype(c), std::function<R(A1, A2)>);

    std::function d{&f2};
    ASSERT_SAME_TYPE(decltype(d), std::function<R(A1, A2)>);
  }
  {
    // implicit
    std::function a = f3;
    ASSERT_SAME_TYPE(decltype(a), std::function<R(A1, A2, A3)>);

    std::function b = &f3;
    ASSERT_SAME_TYPE(decltype(b), std::function<R(A1, A2, A3)>);

    // explicit
    std::function c{f3};
    ASSERT_SAME_TYPE(decltype(c), std::function<R(A1, A2, A3)>);

    std::function d{&f3};
    ASSERT_SAME_TYPE(decltype(d), std::function<R(A1, A2, A3)>);
  }
  // Make sure defaulted arguments don't mess up the deduction
  {
    // implicit
    std::function a = f4;
    ASSERT_SAME_TYPE(decltype(a), std::function<R(A1)>);

    std::function b = &f4;
    ASSERT_SAME_TYPE(decltype(b), std::function<R(A1)>);

    // explicit
    std::function c{f4};
    ASSERT_SAME_TYPE(decltype(c), std::function<R(A1)>);

    std::function d{&f4};
    ASSERT_SAME_TYPE(decltype(d), std::function<R(A1)>);
  }

  return 0;
}
