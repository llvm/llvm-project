//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test reset

#include <memory>
#include <cassert>

#include "unique_ptr_test_helper.h"

int main(int, char**) {
  {
    std::unique_ptr<A> p(new A);
    assert(A::count == 1);
    assert(B::count == 0);
    A* i = p.get();
    assert(i != nullptr);
    p.reset(new B);
    assert(A::count == 1);
    assert(B::count == 1);
  }
  assert(A::count == 0);
  assert(B::count == 0);
  {
    std::unique_ptr<A> p(new B);
    assert(A::count == 1);
    assert(B::count == 1);
    A* i = p.get();
    assert(i != nullptr);
    p.reset(new B);
    assert(A::count == 1);
    assert(B::count == 1);
  }
  assert(A::count == 0);
  assert(B::count == 0);

  return 0;
}
