// Throwing operator new must throw std::bad_alloc on allocation failure
// (here triggered by an oversize request) rather than aborting. Opt-in via
// allocator_may_return_null=1.

// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// Windows ASan can't throw bad_alloc; see
// sanitizer_common/sanitizer_new_handler.h.
// UNSUPPORTED: target={{.*windows.*}}
// TODO: hwasan, lsan, msan and tsan abort instead of throwing; enable them
// once they adopt sanitizer_common/sanitizer_new_operators.inc.
// UNSUPPORTED: hwasan, lsan, msan, tsan
// ubsan does not replace operator new, so whether the oversized request
// fails depends on the system allocator (it succeeds with overcommit).
// UNSUPPORTED: ubsan
// REQUIRES: stable-runtime

// FIXME: Fails after #196388 because libclang_rt.asan links static libc++abi.a.
// XFAIL: android

#include <cstdio>
#include <new>

static const size_t kHugeSize =
#if __LP64__ || defined(_WIN64)
    (1ULL << 40) + 1;
#else
    (3UL << 30) + 1;
#endif

int main() {
  bool caught = false;
  try {
    char *p = new char[kHugeSize];
    fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  } catch (const std::bad_alloc &) {
    caught = true;
  }
  if (caught)
    fprintf(stderr, "caught bad_alloc\n");
  // CHECK: caught bad_alloc
  return 0;
}
