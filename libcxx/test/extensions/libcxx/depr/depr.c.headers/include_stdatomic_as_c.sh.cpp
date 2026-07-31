//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We're building as C, so this test doesn't work when building with modules.
// UNSUPPORTED: clang-modules-build

// GCC complains about unrecognized arguments because we're compiling the
// file as C, but we're passing C++ flags on the command-line.
// UNSUPPORTED: gcc

// Test that stdatomic.h gets the C header with its definitions.

// NOTE: It's not common or recommended to have libc++ in the header search
// path when compiling C files, but it does happen often enough.

// RUN: %{cxx} -c -xc %s -fsyntax-only %{flags} %{compile_flags} -std=c99

#include <stdatomic.h>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  [[maybe_unused]] atomic_bool x;
  return 0;
}
