// RUN: %clangxx_cfi_dso -std=c++11 -g -DSHARED_LIB %s -fPIC -shared -o %t-cfi-so.so
// RUN: %clangxx -std=c++11 -g -DSHARED_LIB %s -fPIC -shared -o %t-nocfi-so.so
// RUN: %clangxx_cfi_dso -std=c++11 -g %s -o %t

// RUN: %expect_crash %t start 2>&1 | FileCheck %s
// RUN: %expect_crash %t mmap 2>&1 | FileCheck %s
// RUN: %expect_crash %t dlopen %t-cfi-so.so 2>&1 | FileCheck %s
// RUN: %expect_crash %t dlclose %t-cfi-so.so 2>&1 | FileCheck %s
// RUN: %expect_crash %t dlopen %t-nocfi-so.so 2>&1 | FileCheck %s
// RUN: %expect_crash %t dlclose %t-nocfi-so.so 2>&1 | FileCheck %s

// Tests that shadow is read-only most of the time.
// REQUIRES: cxxabi

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct A {
  virtual void f();
};

#ifdef SHARED_LIB

void A::f() {}

extern "C" A *create_A() { return new A(); }

#else

constexpr unsigned kShadowGranularity = 12;

namespace __cfi {
uintptr_t GetShadow();
}

void write_shadow(void *ptr) {
  uintptr_t base = __cfi::GetShadow();
  uint16_t *s =
      (uint16_t *)(base + (((uintptr_t)ptr >> kShadowGranularity) << 1));
  fprintf(stderr, "going to crash\n");
  // CHECK: going to crash
  *s = 42;
  fprintf(stderr, "did not crash\n");
  // CHECK-NOT: did not crash
  exit(1);
}

int main(int argc, char *argv[]) {
  assert(argc > 1);
  const bool test_mmap = strcmp(argv[1], "mmap") == 0;
  const bool test_start = strcmp(argv[1], "start") == 0;
  const bool test_dlopen = strcmp(argv[1], "dlopen") == 0;
  const bool test_dlclose = strcmp(argv[1], "dlclose") == 0;
  const char *lib = argc > 2 ? argv[2] : nullptr;

  if (test_start)
    write_shadow((void *)&main);

  if (test_mmap) {
    void *p = mmap(nullptr, 1 << 20, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    assert(p != MAP_FAILED);
    write_shadow((char *)p + 100);
  } else {
    void *handle = dlopen(lib, RTLD_NOW);
    assert(handle);
    void *create_A = dlsym(handle, "create_A");
    assert(create_A);

    if (test_dlopen)
      write_shadow(create_A);

    int res = dlclose(handle);
    assert(res == 0);

    if (test_dlclose)
      write_shadow(create_A);
  }
}
#endif
