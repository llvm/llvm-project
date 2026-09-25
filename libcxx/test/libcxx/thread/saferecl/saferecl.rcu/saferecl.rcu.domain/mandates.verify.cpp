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

#include <rcu>
#include <type_traits>

// template<class T, class D = default_delete<T>>
//   void rcu_retire(T* p, D d = D(), rcu_domain& dom = rcu_default_domain());
// Mandates: is_move_constructible_v<D> is true and the expression d(p) is well-formed.

struct NotDeleter {};

void test(int* ptr) {
  std::rcu_retire(ptr, NotDeleter{});
  // expected-error-re@*:* {{static assertion failed {{.*}}Deleter must be callable with a pointer}}
  // expected-error-re@*:* {{{{.*}}}}
}
