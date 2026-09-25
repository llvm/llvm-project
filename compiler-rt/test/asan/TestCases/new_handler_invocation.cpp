// Throwing operator new must invoke std::new_handler before throwing
// std::bad_alloc, per [new.delete.single]/3 (and /4 for the nothrow form).
// Opt-in via allocator_may_return_null=1: the handler chain runs regardless
// of the flag, but the chain-exhausted action only becomes "throw bad_alloc"
// when the flag is set; otherwise it's ReportOutOfMemory + Die().

// RUN: %clangxx_asan -O0 %s -o %t
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

static int handler_calls = 0;

static void my_handler() {
  ++handler_calls;
  fprintf(stderr, "handler call %d\n", handler_calls);
  // Break the loop. A real handler would free memory and return; this
  // allocation is unrecoverable so we throw to terminate.
  throw std::bad_alloc();
}

int main() {
  std::set_new_handler(my_handler);
  try {
    char *p = new char[kHugeSize];
    fprintf(stderr, "FAIL: allocation unexpectedly returned %p\n", p);
  } catch (const std::bad_alloc &) {
    fprintf(stderr, "caught bad_alloc after %d handler call(s)\n",
            handler_calls);
  }
  // CHECK: handler call 1
  // CHECK: caught bad_alloc after 1 handler call(s)
  return 0;
}
