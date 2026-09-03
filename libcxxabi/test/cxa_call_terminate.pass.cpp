//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Whether the stack unwinds before std::terminate runs for an exception escaping
// noexcept is implementation-defined. Wasm unwinds, so no exception is current
// once terminate runs.
// XFAIL: LIBCXX-WASI-FIXME

// UNSUPPORTED: no-exceptions

// We're testing the diagnosed behaviour here.
// ADDITIONAL_COMPILE_FLAGS: -Wno-exceptions

#include <cassert>
#include <cstdlib>
#include <exception>

#include "test_macros.h"

void func() TEST_NOEXCEPT {
  try {
    throw 1;
  } catch (float) {
  }
}

void terminate_handler() {
  assert(std::current_exception() != nullptr);
  std::exit(0);
}

int main(int, char**) {
  std::set_terminate(terminate_handler);
  func();
  assert(false);
}
