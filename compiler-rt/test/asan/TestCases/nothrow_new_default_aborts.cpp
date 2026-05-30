// With allocator_may_return_null=false (default), nothrow operator new
// aborts on OOM. The handler chain runs (per [new.delete.single]/4); on
// chain exhaustion the runtime emits the asan diagnostic and Die()s rather
// than returning nullptr.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

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
  // No new_handler installed -> chain exhausts immediately -> default flag
  // selects the abort path.
  char *p = new (std::nothrow) char[kHugeSize];
  fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  return 0;
}

// Linux's secondary mmap fails first (out of memory) and Windows's
// kMaxAllowedMallocSize check trips first (requested allocation size); both
// prove the default-flag abort path was taken.
// CHECK: AddressSanitizer: {{out of memory|requested allocation size}}
// CHECK: ABORTING
