// REQUIRES: android

// Tests that ubsan can detect errors on Android if libc appears before the
// runtime in the library search order, which means that we cannot intercept
// symbols.

// RUN: %clangxx %p/Inputs/no-interception-dso.c -fsanitize=undefined -fPIC -shared -o %dynamiclib %ld_flags_rpath_so

// Make sure that libc is first in DT_NEEDED.
// RUN: %clangxx %s -lc -o %t %ld_flags_rpath_exe
// RUN: %run %t 2>&1 | FileCheck %s

#include <limits.h>

int dso_function(int);

int main(int argc, char **argv) {
  // CHECK: signed integer overflow
  dso_function(INT_MAX);
}
