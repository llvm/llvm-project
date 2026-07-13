//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

#include <__stop_token/intrusive_shared_ptr.h>
#include <atomic>
#include <cassert>
#include <utility>

#include "test_macros.h"

struct Object {
  std::atomic<int> counter = 0;
};

template <>
struct std::__intrusive_shared_ptr_traits<Object> {
  static std::atomic<int>& __get_atomic_ref_count(Object& obj) { return obj.counter; }
};

using Ptr = std::__intrusive_shared_ptr<Object>;

int main(int, char**) {
  // default
  {
    Ptr ptr;
    assert(!ptr);
  }

  // raw ptr
  {
    auto object = new Object;
    Ptr ptr(object);
    assert(ptr->counter == 1);
  }

  // copy
  {
    auto object = new Object;
    Ptr ptr(object);
    auto ptr2 = ptr;
    assert(ptr->counter == 2);
    assert(ptr2->counter == 2);
  }

  // move
  {
    auto object = new Object;
    Ptr ptr(object);
    auto ptr2 = std::move(ptr);
    assert(!ptr);
    assert(ptr2->counter == 1);
  }

  // copy assign
  {
    auto object1 = new Object;
    auto object2 = new Object;
    Ptr ptr1(object1);
    Ptr ptr2(object2);

    ptr1 = ptr2;
    assert(ptr1->counter == 2);
    assert(ptr2->counter == 2);
  }

  // move assign
  {
    auto object1 = new Object;
    auto object2 = new Object;
    Ptr ptr1(object1);
    Ptr ptr2(object2);

    ptr1 = std::move(ptr2);
    assert(ptr1->counter == 1);
    assert(!ptr2);
  }

  return 0;
}
