// This test is broken with shared libstdc++ / libc++ on Android.
// RUN: %clangxx_hwasan -static-libstdc++ %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan -static-libstdc++ -DMALLOCEDSTACK %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan -static-libstdc++ -DNO_SANITIZE_F %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan_oldrt -static-libstdc++ %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=GOOD
// RUN: %clangxx_hwasan_oldrt -static-libstdc++ %s -mllvm -hwasan-instrument-landing-pads=0 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=BAD

// C++ tests on x86_64 require instrumented libc++/libstdc++.
// RISC-V target doesn't support oldrt
// REQUIRES: aarch64-target-arch

#include <cassert>
#include <cstdio>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sanitizer/hwasan_interface.h>
#include <stdexcept>
#include <string.h>

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
void *
f(void *) {
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
  return nullptr;
}

int main() {
  __hwasan_enable_allocator_tagging();
#ifdef MALLOCEDSTACK
  pthread_attr_t attr;
  void *stack = malloc(PTHREAD_STACK_MIN);
  assert(pthread_attr_init(&attr) == 0);
  if (pthread_attr_setstack(&attr, stack, PTHREAD_STACK_MIN) != 0) {
    fprintf(stderr, "pthread_attr_setstack: %s", strerror(errno));
    abort();
  }
  pthread_t thid;
  if (pthread_create(&thid, &attr, f, nullptr) != 0) {
    fprintf(stderr, "pthread_create: %s", strerror(errno));
    abort();
  }
  void *ret;
  if (pthread_join(thid, &ret) != 0) {
    fprintf(stderr, "pthread_join: %s", strerror(errno));
    abort();
  }
  assert(pthread_attr_destroy(&attr) == 0);
  free(stack);
#else
  f(nullptr);
#endif
}
