// RUN: %clang %s -o %T/suffix-log-path_test-binary

// The glob below requires bash.
// REQUIRES: shell

// Good log_path with suffix.
// RUN: rm -f %T/sanitizer.log.*.txt
// RUN: %env_tool_opts=log_path=%T/sanitizer.log:log_exe_name=1:log_suffix=.txt %run %T/suffix-log-path_test-binary 2> %t.out
// RUN: FileCheck %s < %T/sanitizer.log.suffix-log-path_test-binary.*.txt

// UNSUPPORTED: ios, android

#include <stdlib.h>
#include <string.h>

#include <sanitizer/common_interface_defs.h>

int main(int argc, char **argv) {
  __sanitizer_print_stack_trace();
  return 0;
}
// CHECK: #{{.*}} main
