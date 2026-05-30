// With allocator_may_return_null=false (default), throwing operator new
// aborts on OOM. The handler chain runs (per [new.delete.single]/3); on
// chain exhaustion the runtime emits the asan ERROR + SUMMARY block and
// Die()s.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

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

int main() {
  // No new_handler installed -> chain exhausts immediately -> default flag
  // selects the abort path.
  char *p = new char[kHugeSize];
  fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  return 0;
}

// CHECK: AddressSanitizer: out of memory
// CHECK: ABORTING
