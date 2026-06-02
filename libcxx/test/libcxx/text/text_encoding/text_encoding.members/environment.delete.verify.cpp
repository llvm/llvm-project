//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// REQUIRES: no-localization

// <text_encoding>

// text_encoding text_encoding::environment()
// template<std::text_encoding::id> text_encoding text_encoding::environment_is()

// environment() and environment_is() are deleted if libc++ is built without localization.

#include <text_encoding>

int main(int, char**) {
  // expected-error@+1 {{attempt to use a deleted function}}
  std::text_encoding::environment();
  // expected-error@+1 {{call to deleted function 'environment_is'}}
  std::text_encoding::environment_is<std::text_encoding::UTF8>();

  return 0;
}
