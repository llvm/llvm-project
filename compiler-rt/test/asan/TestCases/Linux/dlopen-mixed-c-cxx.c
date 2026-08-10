// RUN: %clangxx_asan -xc++ -shared -fPIC -o %t.so - < %s
// RUN: %clang_asan %s -o %t.out -ldl
//
// RUN: env ASAN_OPTIONS=verbosity=1 not %t.out %t.so 2>&1 | FileCheck %s
//
// CHECK: AddressSanitizer: failed to intercept '__cxa_throw'
//
// This tests assumes static linking of the asan runtime.
// UNSUPPORTED: asan-dynamic-runtime

#ifdef __cplusplus

static void foo(void) {
  int i = 0;
  throw(i);
}

extern "C" {
int bar(void);
};
int bar(void) {
  try {
    foo();
  } catch (int i) {
    return i;
  }
  return -1;
}

#else

#include <assert.h>
#include <dlfcn.h>

int main(int argc, char **argv) {
  int (*bar)(void);
  void *handle = dlopen(argv[1], RTLD_LAZY);
  assert(handle);
  bar = dlsym(handle, "bar");
  assert(bar);
  return bar();
}

#endif
