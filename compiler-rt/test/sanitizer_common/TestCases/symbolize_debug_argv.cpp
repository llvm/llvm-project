// RUN: %clangxx %s -g -o %t
// RUN: %env_tool_opts=verbosity=3 %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>

int main(int argc, char **argv) {
  // CHECK: Launching Symbolizer process: {{.+}}
  __sanitizer_print_stack_trace();
  return 0;
}
