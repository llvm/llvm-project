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
    std::unique_ptr<A[]> p;
    p.reset(static_cast<B*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }
  {
    std::unique_ptr<int[]> p;
    p.reset(static_cast<const int*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }

  return 0;
}
