// Aligned array nothrow operator new must return nullptr on allocation
// failure (OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW). Opt-in via
// allocator_may_return_null=1.

// RUN: %clangxx -O0 -std=c++17 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// ASan does not override aligned operator new on Darwin or Windows (MSVC).
// UNSUPPORTED: asan && (darwin || target={{.*windows-msvc.*}})
// ubsan does not replace operator new, so whether the oversized request
// fails depends on the system allocator (it succeeds with overcommit).
// UNSUPPORTED: ubsan
// REQUIRES: stable-runtime

#include <cstdio>
#include <new>

struct alignas(64) HugeAligned {
#if __LP64__ || defined(_WIN64)
  char data[(1ULL << 40) + 1];
#else
  char data[(3UL << 30) + 1];
#endif
};

int main() {
  HugeAligned *p = new (std::nothrow) HugeAligned[1];
  fprintf(stderr, "nothrow aligned array returned %s\n",
          p ? "non-null" : "null");
  // CHECK: nothrow aligned array returned null
  return 0;
}
