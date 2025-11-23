// When DYLD-inserting the ASan dylib from a different location than the
// original, make sure we don't try to reexec.

// UNSUPPORTED: ios

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_asan -print-file-name=lib | tr -d '\n' > %t.lib_name
// RUN: cp %{readfile:%t.lib_name}/darwin/libclang_rt.asan_osx_dynamic.dylib \
// RUN:   %t/libclang_rt.asan_osx_dynamic.dylib
// RUN: %clangxx_asan %s -o %t/a.out

// RUN:   %env_asan_opts=verbosity=1 \
// RUN:       DYLD_INSERT_LIBRARIES=@executable_path/libclang_rt.asan_osx_dynamic.dylib \
// RUN:       %run %t/a.out 2>&1 \
// RUN:   | FileCheck %s

// On OS X 10.10 and lower, if the dylib is not DYLD-inserted, ASan will re-exec.
// RUN: %if mac-os-10-11-or-higher %{ %env_asan_opts=verbosity=1 %run %t/a.out 2>&1 | FileCheck --check-prefix=CHECK-NOINSERT %s %}

// On OS X 10.11 and higher, we don't need to DYLD-insert anymore, and the interceptors
// still installed correctly. Let's just check that things work and we don't try to re-exec.
// RUN: %if mac-os-10-10-or-lower %{ %env_asan_opts=verbosity=1 %run %t/a.out 2>&1 | FileCheck %s %}

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
