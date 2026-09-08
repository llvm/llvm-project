//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// template<class T, class D = default_delete<T>>
//   void rcu_retire(T* p, D d = D(), rcu_domain& dom = rcu_default_domain());

#include <cassert>
#include <rcu>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

class TestClass1 {
  int& destruction_count_;

public:
  TestClass1(int& c) : destruction_count_(c) {}
  ~TestClass1() { ++destruction_count_; }
};

struct Deleter {
  int i = 0;
  template <class T>
  void operator()(T* ptr) const {
    ptr->i_ = i;
    delete ptr;
  }
};

struct CustomDeleter {
  int& i_;
  int& destruction_count_;
  CustomDeleter(int& i, int& destruction_count) : i_(i), destruction_count_(destruction_count) {}
  ~CustomDeleter() { ++destruction_count_; }
};

int main(int, char**) {
  {
    // retire();

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr);
    std::rcu_barrier();
    assert(destruction_count == 1);
  }
  {
    // retire(deleter);
    int i                 = 0;
    int destruction_count = 0;
    auto* ptr             = new CustomDeleter(i, destruction_count);
    std::rcu_retire(ptr, Deleter{42});
    std::rcu_barrier();
    assert(destruction_count == 1);
    assert(i == 42);
  }

  {
    // retire(deleter, domain);
    int i                 = 0;
    int destruction_count = 0;
    auto* ptr             = new CustomDeleter(i, destruction_count);
    std::rcu_retire(ptr, Deleter{42}, std::rcu_default_domain());
    std::rcu_barrier();
    assert(destruction_count == 1);
    assert(i == 42);
  }
  {
    // todo multi threaded test
  }

  return 0;
}
