// RUN: %clangxx_hwasan %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan -DNO_SANITIZE_F %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan_oldrt %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan_oldrt %s -mllvm -hwasan-instrument-landing-pads=0 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=BAD

// C++ tests on x86_64 require instrumented libc++/libstdc++.
// REQUIRES: aarch64-target-arch

#include <stdexcept>
#include <cstdio>

static void optimization_barrier(void* arg) {
  asm volatile("" : : "r"(arg) : "memory");
}

__attribute__((noinline))
void h() {
  char x[1000];
  optimization_barrier(x);
  throw std::runtime_error("hello");
}

__attribute__((noinline))
void g() {
  char x[1000];
  optimization_barrier(x);
  h();
  optimization_barrier(x);
}

__attribute__((noinline))
void hwasan_read(char *p, int size) {
  char volatile sink;
  for (int i = 0; i < size; ++i)
    sink = p[i];
}

__attribute__((noinline, no_sanitize("hwaddress"))) void after_catch() {
  char x[10000];
  hwasan_read(&x[0], sizeof(x));
}


__attribute__((noinline))
#ifdef NO_SANITIZE_F
__attribute__((no_sanitize("hwaddress")))
#endif
void f() {
  char x[1000];
  try {
    // Put two tagged frames on the stack, throw an exception from the deepest one.
    g();
  } catch (const std::runtime_error &e) {
    // Put an untagged frame on stack, check that it is indeed untagged.
    // This relies on exception support zeroing out stack tags.
    // BAD: tag-mismatch
    after_catch();
    // Check that an in-scope stack allocation is still tagged.
    // This relies on exception support not zeroing too much.
    hwasan_read(&x[0], sizeof(x));
    // GOOD: hello
    printf("%s\n", e.what());
  }
}

int main() {
  f();
}
