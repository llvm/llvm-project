// RUN: %clangxx %s -g -O0 -o %t-with-debug

// With debug info atos reports the source location, but no function offset. We fallback to dladdr() to retrieve the function offset.
// RUN: %env_tool_opts=verbosity=2,stack_trace_format='"function_name:%f function_offset:%q"' %run %t-with-debug > %t-with-debug.output 2>&1
// RUN: FileCheck -input-file=%t-with-debug.output %s

// Without debug info atos reports the function offset and so dladdr() fallback is not used.
// RUN: rm -rf %t-with-debug.dSYM
// RUN: %env_tool_opts=verbosity=2,stack_trace_format='"function_name:%f function_offset:%q"' %run %t-with-debug > %t-no-debug.output 2>&1
// RUN: FileCheck -input-file=%t-no-debug.output %s

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

void baz() {
  printf("Do stuff in baz\n");
  __sanitizer_print_stack_trace();
}

void bar() {
  printf("Do stuff in bar\n");
  baz();
}

void foo() {
  printf("Do stuff in foo\n");
  bar();
}

int main() {
  printf("Do stuff in main\n");
  foo();
  return 0;
}

// CHECK: Using atos found at:

// These `function_offset` patterns are designed to disallow `0x0` which is the
// value printed for `kUnknown`.
// CHECK: function_name:baz{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:bar{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:foo{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:main{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
