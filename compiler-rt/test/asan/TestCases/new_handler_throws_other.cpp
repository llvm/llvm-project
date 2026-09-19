// If the registered std::new_handler throws an exception other than
// std::bad_alloc, that exception must propagate out of operator new
// unmodified. Opt-in via allocator_may_return_null=1.

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
  try {
    char *p = new char[kHugeSize];
    fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  } catch (const std::bad_alloc &) {
    fprintf(stderr, "FAIL: caught bad_alloc instead of runtime_error\n");
  } catch (const std::runtime_error &e) {
    fprintf(stderr, "caught runtime_error: %s\n", e.what());
  }
  // CHECK: caught runtime_error: oom-policy
  return 0;
}
