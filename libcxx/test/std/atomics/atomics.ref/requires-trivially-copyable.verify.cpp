//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic_ref>

// template<class T>
// class atomic_ref;

// The program is ill-formed if is_trivially_copyable_v<T> is false.

#include <atomic>

void trivially_copyable() {
  struct X {
    X() = default;
    X(X const&) {} // -> not trivially copyable
  } x;
  // expected-error-re@*:* {{static assertion failed {{.*}}atomic_ref<T> requires that 'T' be a trivially copyable type}}
  std::atomic_ref<X> r(x);
}
