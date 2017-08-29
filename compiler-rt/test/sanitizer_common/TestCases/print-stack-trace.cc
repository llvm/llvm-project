// RUN: %clangxx -O0 %s -o %t && %env_tool_opts=stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O3 %s -o %t && %env_tool_opts=stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=stack_trace_format='"frame:%n lineno:%l"' %run %t 2>&1 | FileCheck %s --check-prefix=CUSTOM
// RUN: %env_tool_opts=symbolize_inline_frames=false:stack_trace_format=DEFAULT %run %t 2>&1 | FileCheck %s --check-prefix=NOINLINE

// UNSUPPORTED: darwin

#include <sanitizer/common_interface_defs.h>

static inline void FooBarBaz() {
  __sanitizer_print_stack_trace();
}

int main() {
  FooBarBaz();
  return 0;
}
// CHECK: {{    #0 0x.* in __sanitizer_print_stack_trace}}
// CHECK: {{    #1 0x.* in FooBarBaz(\(\))? .*}}print-stack-trace.cc:[[@LINE-8]]
// CHECK: {{    #2 0x.* in main.*}}print-stack-trace.cc:[[@LINE-5]]

// CUSTOM: frame:1 lineno:[[@LINE-11]]
// CUSTOM: frame:2 lineno:[[@LINE-8]]

// NOINLINE: #0 0x{{.*}} in __sanitizer_print_stack_trace
// NOINLINE: #1 0x{{.*}} in main{{.*}}print-stack-trace.cc:[[@LINE-15]]
