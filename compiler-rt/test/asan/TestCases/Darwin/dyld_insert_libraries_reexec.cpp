// When DYLD-inserting the ASan dylib from a different location than the
// original, make sure we don't try to reexec.

// UNSUPPORTED: ios

// RUN: rm -rf %t && mkdir -p %t
// RUN: cp `%clang_asan -print-file-name=lib`/darwin/libclang_rt.asan_osx_dynamic.dylib \
// RUN:   %t/libclang_rt.asan_osx_dynamic.dylib
// RUN: %clangxx_asan %s -o %t/a.out

// RUN:   %env_asan_opts=verbosity=1 \
// RUN:       DYLD_INSERT_LIBRARIES=@executable_path/libclang_rt.asan_osx_dynamic.dylib \
// RUN:       %run %t/a.out 2>&1 \
// RUN:   | FileCheck %s

// RUN: IS_OSX_10_11_OR_HIGHER=$([ `sw_vers -productVersion | cut -d'.' -f2` -lt 11 ]; echo $?)

// On OS X 10.10 and lower, if the dylib is not DYLD-inserted, ASan will re-exec.
// RUN: if [ $IS_OSX_10_11_OR_HIGHER == 0 ]; then \
// RUN:   %env_asan_opts=verbosity=1 %run %t/a.out 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOINSERT %s; \
// RUN:   fi

// On OS X 10.11 and higher, we don't need to DYLD-insert anymore, and the interceptors
// still installed correctly. Let's just check that things work and we don't try to re-exec.
// RUN: if [ $IS_OSX_10_11_OR_HIGHER == 1 ]; then \
// RUN:   %env_asan_opts=verbosity=1 %run %t/a.out 2>&1 \
// RUN:   | FileCheck %s; \
// RUN:   fi

#include <stdio.h>

int main() {
  printf("Passed\n");
  return 0;
}

// CHECK-NOINSERT: exec()-ing the program with
// CHECK-NOINSERT: DYLD_INSERT_LIBRARIES
// CHECK-NOINSERT: to enable wrappers.
// CHECK-NOINSERT: Passed

// CHECK-NOT: exec()-ing the program with
// CHECK-NOT: DYLD_INSERT_LIBRARIES
// CHECK-NOT: to enable wrappers.
// CHECK: Passed
