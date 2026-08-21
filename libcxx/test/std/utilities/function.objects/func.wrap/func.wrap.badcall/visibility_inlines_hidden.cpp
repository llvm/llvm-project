//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -fvisibility-inlines-hidden

// When there is a weak hidden symbol in user code and a strong definition
// in the library, we test that the linker relies on the library version,
// as the default weak resolution semantics don't favour weak local definitions
// for XCOFF. This creates a conflict on std::bad_function_call, which is used
// by the std::function template instantiated in main.
#include <functional>
#include "test_macros.h"
#include "assert.h"

void foo() {}

void test_call() {
  std::function<void()> r(foo);
  r();
}

void test_throw() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  std::function<int()> f;
  try {
    f();
    assert(false);
  } catch (const std::bad_function_call&) {
    return;
  }
  assert(false);
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test_call();
  test_throw();
  return 0;
}
