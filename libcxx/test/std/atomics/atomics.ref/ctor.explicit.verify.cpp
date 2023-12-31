//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// explicit atomic_ref(T&);

#include <atomic>

template <class T>
void test(std::atomic_ref<T>) {}

void explicit_ctor() {
  int i = 0;
  // expected-error-re@*:* {{{{.*}}no matching function for call to 'test'}}
  test<int>(i);

  float f = 0.f;
  // expected-error-re@*:* {{{{.*}}no matching function for call to 'test'}}
  test<float>(f);

  int* p = &i;
  // expected-error-re@*:* {{{{.*}}no matching function for call to 'test'}}
  test<int*>(p);

  struct X {
  } x;
  // expected-error-re@*:* {{{{.*}}no matching function for call to 'test'}}
  test<X>(x);
}
