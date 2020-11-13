// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We're building as C, so this test doesn't work when building with modules.
// UNSUPPORTED: -fmodules

// GCC complains about unrecognized arguments because we're compiling the
// file as C, but we're passing C++ flags on the command-line.
// UNSUPPORTED: gcc

// Test that the C wrapper headers can be included when compiling them as C.

// NOTE: It's not common or recommended to have libc++ in the header search
// path when compiling C files, but it does happen often enough.

// RUN: %{cxx} -c -xc %s -fsyntax-only %{flags} %{compile_flags} -std=c99

#include <__config>

#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#   include <locale.h>
#endif
#include <math.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <wchar.h>
#include <wctype.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}
