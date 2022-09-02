//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test get_deleter()

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

struct Deleter {
  Deleter() {}

  void operator()(void*) const {}

  int test() { return 5; }
  int test() const { return 6; }
};

template <bool IsArray>
void test_basic() {
  typedef typename std::conditional<IsArray, int[], int>::type VT;
  {
    std::unique_ptr<int, Deleter> p;
    assert(p.get_deleter().test() == 5);
  }
  {
    const std::unique_ptr<VT, Deleter> p;
    assert(p.get_deleter().test() == 6);
  }
  {
    typedef std::unique_ptr<VT, const Deleter&> UPtr;
    const Deleter d;
    UPtr p(nullptr, d);
    const UPtr& cp = p;
    ASSERT_SAME_TYPE(decltype(p.get_deleter()), const Deleter&);
    ASSERT_SAME_TYPE(decltype(cp.get_deleter()), const Deleter&);
    assert(p.get_deleter().test() == 6);
    assert(cp.get_deleter().test() == 6);
  }
  {
    typedef std::unique_ptr<VT, Deleter&> UPtr;
    Deleter d;
    UPtr p(nullptr, d);
    const UPtr& cp = p;
    ASSERT_SAME_TYPE(decltype(p.get_deleter()), Deleter&);
    ASSERT_SAME_TYPE(decltype(cp.get_deleter()), Deleter&);
    assert(p.get_deleter().test() == 5);
    assert(cp.get_deleter().test() == 5);
  }
}

int main(int, char**) {
  test_basic</*IsArray*/ false>();
  test_basic<true>();

  return 0;
}
