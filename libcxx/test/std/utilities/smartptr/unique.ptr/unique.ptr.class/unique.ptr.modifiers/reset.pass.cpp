//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test reset

#include <memory>
#include <cassert>

#include "unique_ptr_test_helper.h"

template <bool IsArray>
void test_basic() {
  typedef typename std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
  {
    std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    assert(A::count == expect_alive);
    A* i = p.get();
    assert(i != nullptr);
    p.reset();
    assert(A::count == 0);
    assert(p.get() == 0);
  }
  assert(A::count == 0);
  {
    std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    assert(A::count == expect_alive);
    A* i = p.get();
    assert(i != nullptr);
    A* new_value = newValue<VT>(expect_alive);
    assert(A::count == (expect_alive * 2));
    p.reset(new_value);
    assert(A::count == expect_alive);
  }
  assert(A::count == 0);
}

int main() {
  test_basic</*IsArray*/ false>();
  test_basic<true>();
}
