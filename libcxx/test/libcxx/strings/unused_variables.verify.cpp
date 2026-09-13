//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we warn on unused variables of libc++ classes which behave like value types.
// ADDITIONAL_COMPILE_FLAGS: -Wunused-variable

#include <string>

void strings() {
  std::string l;                 // expected-warning {{unused variable}}
  std::string::iterator l;       // expected-warning {{unused variable}}
  std::string::const_iterator l; // expected-warning {{unused variable}}
}
