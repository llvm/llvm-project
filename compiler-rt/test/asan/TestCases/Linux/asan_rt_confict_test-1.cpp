// Test that preloading dynamic runtime to statically sanitized
// executable is prohibited.
//
// RUN: %clangxx_asan_static %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime
// XFAIL: android

#include <stdlib.h>
int main(int argc, char **argv) { return 0; }

// CHECK: Your application is linked against incompatible ASan runtimes
