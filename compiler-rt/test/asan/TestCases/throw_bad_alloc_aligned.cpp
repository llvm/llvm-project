// Throwing aligned operator new must throw std::bad_alloc on allocation
// failure, just like the unaligned form. Opt-in via allocator_may_return_null=1.

// RUN: %clangxx_asan -O0 -std=c++17 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: target={{.*windows.*}}
// REQUIRES: stable-runtime

#include <cstdio>
#include <new>

static const size_t kHugeSize =
#if __LP64__ || defined(_WIN64)
    (1ULL << 40) + 1;
#else
    (3UL << 30) + 1;
#endif

struct alignas(64) Aligned {
  char data[1];
};

int main() {
  bool caught = false;
  try {
    Aligned *p = new Aligned[kHugeSize];
    fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  } catch (const std::bad_alloc &) {
    caught = true;
  }
  if (caught)
    fprintf(stderr, "caught bad_alloc\n");
  // CHECK: caught bad_alloc
  return 0;
}
