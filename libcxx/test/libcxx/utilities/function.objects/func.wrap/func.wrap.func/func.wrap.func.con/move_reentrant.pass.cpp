//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// function& operator=(function &&);

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX03_FUNCTION

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
  static std::function<void()> global;
  static bool cancel;

  ~A() {
    DoNotOptimize(cancel);
    if (cancel)
      global = std::function<void()>(nullptr);
  }
  void operator()() {}
};

std::function<void()> A::global;
bool A::cancel = false;

int main(int, char**)
{
  A::global = A();
  RTTI_ASSERT(A::global.target<A>());

  // Check that we don't recurse in A::~A().
  A::cancel = true;
  A::global = std::function<void()>(nullptr);
  RTTI_ASSERT(!A::global.target<A>());

  return 0;
}
