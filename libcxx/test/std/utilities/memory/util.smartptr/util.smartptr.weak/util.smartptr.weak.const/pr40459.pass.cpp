//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// weak_ptr

// template<class Y> weak_ptr(const weak_ptr<Y>& r);
// template<class Y> weak_ptr(weak_ptr<Y>&& r);
//
// Regression test for https://llvm.org/PR41114
// Verify that these constructors never attempt a derived-to-virtual-base
// conversion on a dangling weak_ptr.

#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

#include "test_macros.h"

struct A {
  int i;
  virtual ~A() {}
};
struct B : public virtual A {
  int j;
};
struct Deleter {
  void operator()(void*) const {
    // do nothing
  }
};

int main(int, char**) {
#if TEST_STD_VER >= 11
  alignas(B) char buffer[sizeof(B)];
#else
  std::aligned_storage<sizeof(B), std::alignment_of<B>::value>::type buffer;
#endif
  B* pb                 = ::new ((void*)&buffer) B();
  std::shared_ptr<B> sp = std::shared_ptr<B>(pb, Deleter());
  std::weak_ptr<B> wp   = sp;
  sp                    = nullptr;
  assert(wp.expired());

  // Overwrite the B object with junk.
  std::memset(&buffer, '*', sizeof(buffer));

  std::weak_ptr<A> wq = wp;
  assert(wq.expired());
  std::weak_ptr<A> wr = std::move(wp);
  assert(wr.expired());

  return 0;
}
