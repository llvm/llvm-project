// Checks that on OS X 10.11+ (where we do not re-exec anymore, because
// interceptors work automatically), dlopen'ing a TSanified library from a
// non-instrumented program exits with a user-friendly message.

// REQUIRES: osx-autointerception

// XFAIL: ios

// RUN: %clangxx_tsan %s -o %t.so -shared -DSHARED_LIB
// RUN: %clangxx_tsan -fno-sanitize=thread %s -o %t

// RUN: TSAN_DYLIB_PATH=`%clangxx_tsan %s -### 2>&1 \
// RUN:   | grep "libclang_rt.tsan_osx_dynamic.dylib" \
// RUN:   | sed -e 's/.*"\(.*libclang_rt.tsan_osx_dynamic.dylib\)".*/\1/'`

// Launching a non-instrumented binary that dlopen's an instrumented library should fail.
// RUN: not %run %t %t.so 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// Launching a non-instrumented binary with an explicit DYLD_INSERT_LIBRARIES should work.
// RUN: DYLD_INSERT_LIBRARIES=$TSAN_DYLIB_PATH %run %t %t.so 2>&1 | FileCheck %s

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

#if defined(SHARED_LIB)
extern "C" void foo() {
  fprintf(stderr, "Hello world.\n");
}
#else  // defined(SHARED_LIB)
int main(int argc, char *argv[]) {
  void *handle = dlopen(argv[1], RTLD_NOW);
  fprintf(stderr, "handle = %p\n", handle);
  void (*foo)() = (void (*)())dlsym(handle, "foo");
  fprintf(stderr, "foo = %p\n", foo);
  foo();
}
#endif  // defined(SHARED_LIB)

// CHECK: Hello world.
// CHECK-NOT: ERROR: Interceptors are not working.

// CHECK-FAIL-NOT: Hello world.
// CHECK-FAIL: ERROR: Interceptors are not working.
