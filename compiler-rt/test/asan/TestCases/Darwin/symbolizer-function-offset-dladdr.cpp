// FIXME(dliew): Duplicated from `test/sanitizer_common/TestCases/Darwin/symbolizer-function-offset-dladdr.cpp`.
// This case can be dropped once sanitizer_common tests work on iOS devices (rdar://problem/47333049).

// NOTE: `detect_leaks=0` is necessary because with LSan enabled the dladdr
// symbolizer actually leaks memory because the call to
// `__sanitizer::DemangleCXXABI` leaks memory which LSan detects
// (rdar://problem/42868950).

// RUN: %clangxx_asan %s -O0 -o %t
// RUN: %env_asan_opts=detect_leaks=0,verbosity=2,external_symbolizer_path=,stack_trace_format='"function_name_%f___function_offset_%q"' %run %t > %t.output 2>&1
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
// CHECK: function_name_baz{{(\(\))?}}___function_offset_0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name_bar{{(\(\))?}}___function_offset_0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name_foo{{(\(\))?}}___function_offset_0x{{0*[1-9a-f][0-9a-f]*$}}
// CHECK: function_name_main{{(\(\))?}}___function_offset_0x{{0*[1-9a-f][0-9a-f]*$}}
