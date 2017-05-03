// Check that when launching with DYLD_INSERT_LIBRARIES, we properly remove
// the ASan dylib from the environment variable (both when using an absolute
// or relative path) and also that the other dylibs are left untouched.

// UNSUPPORTED: ios

// RUN: mkdir -p %T/dyld_insert_libraries_remove
// RUN: cp `%clang_asan %s -fsanitize=address -### 2>&1 \
// RUN:   | grep "libclang_rt.asan_osx_dynamic.dylib" \
// RUN:   | sed -e 's/.*"\(.*libclang_rt.asan_osx_dynamic.dylib\)".*/\1/'` \
// RUN:   %T/dyld_insert_libraries_remove/libclang_rt.asan_osx_dynamic.dylib

// RUN: %clangxx_asan %s -o %T/dyld_insert_libraries_remove/a.out
// RUN: %clangxx -DSHARED_LIB %s \
// RUN:     -dynamiclib -o %T/dyld_insert_libraries_remove/dummy-so.dylib

// RUN: ( cd %T/dyld_insert_libraries_remove && \
// RUN:   DYLD_INSERT_LIBRARIES=@executable_path/libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
// RUN:   %run ./a.out 2>&1 ) | FileCheck %s || exit 1

// RUN: ( cd %T/dyld_insert_libraries_remove && \
// RUN:   DYLD_INSERT_LIBRARIES=libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
// RUN:   %run ./a.out 2>&1 ) | FileCheck %s || exit 1

// RUN: ( cd %T/dyld_insert_libraries_remove && \
// RUN:   DYLD_INSERT_LIBRARIES=%T/dyld_insert_libraries_remove/libclang_rt.asan_osx_dynamic.dylib:dummy-so.dylib \
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
