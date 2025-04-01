// RUN: %clangxx -fsanitize=undefined -O0 %s -o %t && UBSAN_OPTIONS=stack_trace_format=DEFAULT:fast_unwind_on_fatal=1 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=undefined -O0 %s -o %t && UBSAN_OPTIONS=stack_trace_format=DEFAULT:fast_unwind_on_fatal=0 %run %t 2>&1 | FileCheck %s

// This test is temporarily disabled due to broken unwinding on ARM.
// UNSUPPORTED: target={{.*-linux-.*}}

// The test doesn't pass on Darwin in UBSan-TSan configuration, because TSan is
// using the slow unwinder which is not supported on Darwin. The test should
// be universal after landing of https://reviews.llvm.org/D32806.

#include <sanitizer/common_interface_defs.h>

static inline void FooBarBaz() {
  __sanitizer_print_stack_trace();
}

int main() {
  FooBarBaz();
  return 0;
}

// CHECK: {{.*}} in FooBarBaz{{.*}}print_stack_trace.cpp{{.*}}
// CHECK: {{.*}} in main{{.*}}print_stack_trace.cpp{{.*}}
