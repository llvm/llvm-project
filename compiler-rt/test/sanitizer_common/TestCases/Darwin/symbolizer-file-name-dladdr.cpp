// RUN: %clangxx %s -O0 -o %t
// RUN: %env_tool_opts=external_symbolizer_path= %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

int main() {
  //CHECK: #0{{.*}}
  //CHECK: #1{{.*}}(symbolizer-file-name-dladdr.cpp.tmp
  __sanitizer_print_stack_trace();
  return 0;
}
