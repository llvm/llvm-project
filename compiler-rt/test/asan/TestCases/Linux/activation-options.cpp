// Test for ASAN_OPTIONS=start_deactivated=1 mode.
// Main executable is uninstrumented, but linked to ASan runtime. The shared
// library is instrumented.
// Fails with debug checks: https://bugs.llvm.org/show_bug.cgi?id=46862
// XFAIL: !compiler-rt-optimized

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx -O0 %s -c -o %t.o
// RUN: %clangxx_asan -O0 %t.o %libdl -o %t

// RUN: rm -f %t.asan.options.activation-options.cpp.tmp
// RUN: rm -f %t.asan.options.ABCDE
// RUN: echo "help=1" >%t.asan.options.activation-options.cpp.tmp

// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=include=%t.asan.options.%b %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-HELP --check-prefix=CHECK-FOUND

// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=include=%t.asan.options not %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-HELP --check-prefix=CHECK-MISSING

// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=include=%t.asan.options.%b not %run %t --fix-name 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-HELP --check-prefix=CHECK-MISSING

// RUN: echo "help=1" >%t.asan.options.ABCDE

// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=include=%t.asan.options.%b %run %t --fix-name 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-HELP --check-prefix=CHECK-FOUND

// XFAIL: android

#if !defined(SHARED_LIB)
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>

#include "sanitizer/asan_interface.h"

typedef void (*Fn)();

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + "-so.so";

  if (argc > 1 && strcmp(argv[1], "--fix-name") == 0) {
    assert(strlen(argv[0]) > 5);
    strcpy(argv[0], "ABCDE");
  }

  void *dso = dlopen(path.c_str(), RTLD_NOW);
  if (!dso) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  return 0;
}
#else  // SHARED_LIB
// Empty: all we need is an ASan shared library constructor.
#endif  // SHARED_LIB

// CHECK-HELP: Available flags for {{.*}}Sanitizer:
// CHECK-NO-HELP-NOT: Available flags for {{.*}}Sanitizer:
// CHECK-FOUND-NOT: Failed to read options
// CHECK-MISSING: Failed to read options
