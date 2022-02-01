// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/ %run %t 2>&1 | FileCheck %s
//
// Tests __sanitizer_symbolize_pc.
#include <stdio.h>
#include <sanitizer/common_interface_defs.h>

int GLOBAL_VAR_ABC;

void SymbolizeSmallBuffer() {
  char data[] = "abcdef";
  __sanitizer_symbolize_pc(__sanitizer_return_address(), "%p %F %L", data, 0);
  printf("UNCHANGED '%s'\n", data);
  __sanitizer_symbolize_pc(__sanitizer_return_address(), "%p %F %L", data, 1);
  printf("EMPTY '%s'\n", data);
  __sanitizer_symbolize_pc(__sanitizer_return_address(), "%p %F %L", data,
                           sizeof(data));
  printf("PARTIAL '%s'\n", data);
}

void SymbolizeCaller() {
  char data[100];
  __sanitizer_symbolize_pc(__sanitizer_return_address(), "%p %F %L", data,
                           sizeof(data));
  printf("FIRST_FORMAT %s\n", data);
  __sanitizer_symbolize_pc(__sanitizer_return_address(),
                           "FUNC:%f LINE:%l FILE:%s", data, sizeof(data));
  printf("SECOND_FORMAT %s\n", data);
  __sanitizer_symbolize_pc(__sanitizer_return_address(),
                           "LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                           "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                           "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                           "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                           "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG"
                           "FUNC:%f LINE:%l FILE:%s",
                           data, sizeof(data));
  printf("LONG_FORMAT %s\n", data);
}

struct s {
  int i;
};

struct s SymbolizeSRet() {
  char data[100];
  __sanitizer_symbolize_pc(__sanitizer_return_address(),
                           "FUNC:%f LINE:%l FILE:%s", data, sizeof(data));
  printf("SRET: %s\n", data);
  struct s s = {1};
  return s;
}

void SymbolizeData() {
  char data[100];
  __sanitizer_symbolize_global(&GLOBAL_VAR_ABC, "%g %s:%l", data, sizeof(data));
  printf("GLOBAL: %s\n", data);
}

int main() {
  // CHECK: UNCHANGED 'abcdef'
  // CHECK: EMPTY ''
  // CHECK: PARTIAL '0x{{.*}}'
  SymbolizeSmallBuffer();

  // CHECK: FIRST_FORMAT 0x{{.*}} in main symbolize_pc.cpp:[[@LINE+2]]
  // CHECK: SECOND_FORMAT FUNC:main LINE:[[@LINE+1]] FILE:symbolize_pc.cpp
  SymbolizeCaller();

  struct s s;
  // CHECK: SRET: FUNC:main LINE:[[@LINE+1]] FILE:symbolize_pc.cpp
  s = SymbolizeSRet();

  // CHECK: GLOBAL: GLOBAL_VAR_ABC
  SymbolizeData();
}
