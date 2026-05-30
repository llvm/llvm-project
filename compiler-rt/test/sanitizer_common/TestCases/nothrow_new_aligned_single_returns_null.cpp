// Aligned single-object nothrow operator new must return nullptr on
// allocation failure (OPERATOR_NEW_BODY_ALIGN_NOTHROW). Opt-in via
// allocator_may_return_null=1.

// RUN: %clangxx -O0 -std=c++17 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s

// ASan does not override aligned operator new on Darwin or Windows (MSVC).
// UNSUPPORTED: asan && (darwin || target={{.*windows-msvc.*}})
// REQUIRES: operator-new-framework, stable-runtime

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
  HugeAligned *p = new (std::nothrow) HugeAligned;
  fprintf(stderr, "nothrow aligned returned %s\n", p ? "non-null" : "null");
  // CHECK: nothrow aligned returned null
  return 0;
}
