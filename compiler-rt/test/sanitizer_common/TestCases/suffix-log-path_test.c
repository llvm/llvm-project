// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: %clang %s -o %t.dir/suffix-log-path_test-binary

// Good log_path with suffix.
// RUN: %env_tool_opts=log_path=%t.dir/sanitizer.log:log_exe_name=1:log_suffix=.txt %run %t.dir/suffix-log-path_test-binary 2> %t.out
// RUN: FileCheck %s < %t.dir/sanitizer.log.suffix-log-path_test-binary.*.txt

// UNSUPPORTED: ios, android

#include <stdlib.h>
#include <string.h>

#include <sanitizer/common_interface_defs.h>

int main(int argc, char **argv) {
  __sanitizer_print_stack_trace();
  return 0;
}
// CHECK: #{{.*}} main
