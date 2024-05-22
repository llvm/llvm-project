//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// function& operator=(nullptr_t);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
  static std::function<void()> global;
  static bool cancel;

  A() = default;
  A(const A&) = default;
  A& operator=(const A&) = default;
  ~A() {
    DoNotOptimize(cancel);
    if (cancel)
      global = nullptr;
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
  A::global = nullptr;
  RTTI_ASSERT(!A::global.target<A>());

  return 0;
}
