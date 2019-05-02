// RUN: %clangxx  -O0 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime && !ubsan && !darwin

#include <stdio.h>

extern "C" void *reallocarray(void *, size_t, size_t);

int main() {
  void *p = reallocarray(nullptr, -1, 1000);
  // CHECK: {{ERROR: .*Sanitizer: reallocarray parameters overflow: count \* size \(.* \* 1000\) cannot be represented in type size_t}}

  printf("reallocarray returned: %zu\n", (size_t)p);
  // CHECK-NULL: reallocarray returned: 0

  return 0;
}
