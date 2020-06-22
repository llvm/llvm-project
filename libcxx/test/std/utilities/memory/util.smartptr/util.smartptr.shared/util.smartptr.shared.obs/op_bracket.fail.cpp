//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14

// shared_ptr

// element_type& operator[](ptrdiff_t i) const;

#include "test_macros.h"

#include <memory>
#include <cassert>

int main(int, char**) {
  // Check that we get a static assertion when we try to use the bracket
  // operator on shared_ptr<T> when T is not an array type.
  const std::shared_ptr<int> p;
  (void)p
      [0]; // expected-error@memory:* {{std::shared_ptr<T>::operator[] is only valid when T is an array type.}}
  return 0;
}
