//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// void fill(const T& u);

#include <array>
#include <cassert>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main() {
  {
    typedef double T;
    typedef std::array<const T, 0> C;
    C c = {};
    // expected-error-re@array:* {{static_assert failed {{.*}}"cannot fill zero-sized array of type 'const T'"}}
    c.fill(5.5); // expected-note {{requested here}}
  }
}
