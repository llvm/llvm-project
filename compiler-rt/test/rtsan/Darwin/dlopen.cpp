// Checks that on OS X 10.11+ dlopen'ing a RTsanified library from a
// non-instrumented program exits with a user-friendly message.

// REQUIRES: osx-autointerception

// XFAIL: ios

// RUN: %clangxx -fsanitize=realtime %s -o %t.so -shared -DSHARED_LIB
// RUN: %clangxx %s -o %t

// RUN: RTSAN_DYLIB_PATH=`%clangxx -fsanitize=realtime %s -### 2>&1 \
// RUN:   | grep "libclang_rt.rtsan_osx_dynamic.dylib" \
// RUN:   | sed -e 's/.*"\(.*libclang_rt.rtsan_osx_dynamic.dylib\)".*/\1/'`

// Launching a non-instrumented binary that dlopen's an instrumented library should fail.
// RUN: not %run %t %t.so 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// Launching a non-instrumented binary with an explicit DYLD_INSERT_LIBRARIES should work.
// RUN: DYLD_INSERT_LIBRARIES=$RTSAN_DYLIB_PATH %run %t %t.so 2>&1 | FileCheck %s

// Launching an instrumented binary with the DYLD_INSERT_LIBRARIES env variable has no error
// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: DYLD_INSERT_LIBRARIES=$RTSAN_DYLIB_PATH %run %t %t.so 2>&1 | FileCheck %s --check-prefix=CHECK-INSTRUMENTED

#include <dlfcn.h>
#include <stdio.h>

#if defined(SHARED_LIB)
extern "C" void foo() { fprintf(stderr, "Hello world.\n"); }
#else  // defined(SHARED_LIB)
int main(int argc, char *argv[]) {
  void *handle = dlopen(argv[1], RTLD_NOW);
  void (*foo)() = (void (*)())dlsym(handle, "foo");
  foo();
}
#endif // defined(SHARED_LIB)

// CHECK: Hello world.
// CHECK-NOT: ERROR: Interceptors are not working.

// CHECK-FAIL-NOT: Hello world.
// CHECK-FAIL: ERROR: Interceptors are not working.

// CHECK-INSTRUMENTED-NOT: ERROR: Interceptors are not working
// CHECK-INSTRUMENTED: Hello world.
