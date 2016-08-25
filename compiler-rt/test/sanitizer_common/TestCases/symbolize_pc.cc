// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/ %run %t 2>&1 | FileCheck %s
//
// Tests __sanitizer_symbolize_pc.
#include <stdio.h>
#include <sanitizer/common_interface_defs.h>
void SymbolizeCaller() {
  char data[1000];
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", data,
                           sizeof(data));
  printf("FIRST_FORMAT %s\n", data);
  __sanitizer_symbolize_pc(__builtin_return_address(0),
                           "FUNC:%f LINE:%l FILE:%s", data, sizeof(data));
  printf("SECOND_FORMAT %s\n", data);
}

// CHECK: FIRST_FORMAT 0x{{.*}} in main symbolize_pc.cc:[[@LINE+3]]
// CHECK: SECOND_FORMAT FUNC:main LINE:[[@LINE+2]] FILE:symbolize_pc.cc
int main() {
  SymbolizeCaller();
}
