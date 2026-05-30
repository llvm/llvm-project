// If the registered std::new_handler throws an exception other than
// std::bad_alloc, the nothrow operator new must catch it and return nullptr
// (per [new.delete.single]/4 which specifies as-if try/catch behavior; the
// framework's try/catch swallows all exception types, not just bad_alloc).
// Opt-in via allocator_may_return_null=1.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: target={{.*windows.*}}
// REQUIRES: stable-runtime

#include <cstdio>
#include <new>
#include <stdexcept>

static const size_t kHugeSize =
#if __LP64__ || defined(_WIN64)
    (1ULL << 40) + 1;
#else
    (3UL << 30) + 1;
#endif

static void my_handler() { throw std::runtime_error("oom-policy"); }

int main() {
  std::set_new_handler(my_handler);
  char *p = new (std::nothrow) char[kHugeSize];
  fprintf(stderr, "nothrow returned %s\n", p ? "non-null" : "null");
  // CHECK: nothrow returned null
  return 0;
}
