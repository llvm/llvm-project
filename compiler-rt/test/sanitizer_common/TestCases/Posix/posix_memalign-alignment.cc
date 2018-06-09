// RUN: %clangxx %collect_stack_traces -O0 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 17 2>&1 | FileCheck %s
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 0 2>&1 | FileCheck %s
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 17 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

// UNSUPPORTED: tsan, ubsan

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  const int alignment = atoi(argv[1]);

  void *p = reinterpret_cast<void*>(42);

  int res = posix_memalign(&p, alignment, 100);
  // CHECK: {{ERROR: .*Sanitizer: invalid alignment requested in posix_memalign}}
  // CHECK: {{#0 0x.* in .*posix_memalign}}
  // CHECK: {{#1 0x.* in main .*posix_memalign-alignment.cc:}}[[@LINE-3]]
  // CHECK: {{SUMMARY: .*Sanitizer: invalid-posix-memalign-alignment}}

  printf("pointer after failed posix_memalign: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed posix_memalign: 42

  return 0;
}
