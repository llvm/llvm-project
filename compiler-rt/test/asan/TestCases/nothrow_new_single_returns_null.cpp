// Single-object nothrow operator new must return nullptr on allocation
// failure (OPERATOR_NEW_BODY_NOTHROW). Opt-in via allocator_may_return_null=1.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <cstdio>
#include <new>

struct alignas(1) Huge {
#if __LP64__ || defined(_WIN64)
  char data[(1ULL << 40) + 1];
#else
  char data[(3UL << 30) + 1];
#endif
};

int main() {
  Huge *p = new (std::nothrow) Huge;
  fprintf(stderr, "nothrow returned %s\n", p ? "non-null" : "null");
  // CHECK: nothrow returned null
  return 0;
}
