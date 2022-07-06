//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// void swap(array& a);

#include <array>
#include <cassert>

int main(int, char**) {
  {
    typedef double T;
    typedef std::array<const T, 0> C;
    C c = {};
    C c2 = {};
    // expected-error-re@array:* {{static_assert failed{{.*}}cannot swap zero-sized array of type 'const T'}}
    c.swap(c2); // expected-note {{requested here}}
  }

  return 0;
}
