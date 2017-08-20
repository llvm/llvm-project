// Check that when launching with DYLD_INSERT_LIBRARIES, we properly remove
// the ASan dylib from the environment variable (both when using an absolute
// or relative path) and also that the other dylibs are left untouched.

// UNSUPPORTED: ios

// RUN: rm -rf %t && mkdir -p %t
// RUN: cp `%clang_asan -print-file-name=lib`/darwin/libclang_rt.asan_osx_dynamic.dylib \
// RUN:   %t/libclang_rt.asan_osx_dynamic.dylib

// RUN: %clangxx_asan %s -o %t/a.out
// RUN: %clangxx -DSHARED_LIB %s \
// RUN:     -dynamiclib -o %t/dummy-so.dylib

// RUN: ( cd %t && \
// RUN:   DYLD_INSERT_LIBRARIES=@executable_path/libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
// RUN:   %run ./a.out 2>&1 ) | FileCheck %s || exit 1

// RUN: ( cd %t && \
// RUN:   DYLD_INSERT_LIBRARIES=libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
// RUN:   %run ./a.out 2>&1 ) | FileCheck %s || exit 1

// RUN: ( cd %t && \
// RUN:   DYLD_INSERT_LIBRARIES=%t/libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
// RUN:   %run ./a.out 2>&1 ) | FileCheck %s || exit 1

#if !defined(SHARED_LIB)
#include <stdio.h>
#include <stdlib.h>

int main() {
  const char kEnvName[] = "DYLD_INSERT_LIBRARIES";
  printf("%s=%s\n", kEnvName, getenv(kEnvName));
  // CHECK: {{DYLD_INSERT_LIBRARIES=dummy-so.dylib}}
  return 0;
}
#else  // SHARED_LIB
void foo() {}
#endif  // SHARED_LIB
