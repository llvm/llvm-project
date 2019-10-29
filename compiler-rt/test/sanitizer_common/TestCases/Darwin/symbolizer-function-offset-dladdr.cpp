// UNSUPPORTED: lsan
// This test fails with LSan enabled because the dladdr symbolizer actually leaks
// memory because the call to `__sanitizer::DemangleCXXABI` leaks memory which LSan
// detects (rdar://problem/42868950).

// RUN: %clangxx %s -O0 -o %t
// RUN: %env_tool_opts=verbosity=2,external_symbolizer_path=,stack_trace_format='"function_name:%f function_offset:%q"' %run %t > %t.output 2>&1
// RUN: FileCheck -input-file=%t.output %s
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

// CHECK: External symbolizer is explicitly disabled
// CHECK: Using dladdr symbolizer

// These `function_offset` patterns are designed to disallow `0x0` which is the
// value printed for `kUnknown`.
// CHECK: function_name:baz{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:bar{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:foo{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name:main{{(\(\))?}} function_offset:0x{{0*[1-9a-f][0-9a-f]*$}}
