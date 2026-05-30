// Per [new.delete.single]/4, nothrow operator new must return nullptr on
// allocation failure after running the new_handler chain. Opt-in via
// allocator_may_return_null=1; the default-flag abort case is covered by
// nothrow_new_default_aborts.cpp.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <cstdio>
#include <new>

static const size_t kHugeSize =
#if __LP64__ || defined(_WIN64)
    (1ULL << 40) + 1;
#else
    (3UL << 30) + 1;
#endif

int main() {
  char *p = new (std::nothrow) char[kHugeSize];
  fprintf(stderr, "nothrow returned %s\n", p ? "non-null" : "null");
  // CHECK: nothrow returned null
  return 0;
}
