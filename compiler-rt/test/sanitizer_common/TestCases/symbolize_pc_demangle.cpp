// RUN: %clangxx -O1 -fno-omit-frame-pointer %s -o %t
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/ %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/:demangle=0 %run %t 2>&1 | FileCheck %s --check-prefixes=NODEMANGLE
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/:demangle=1 %run %t 2>&1 | FileCheck %s

// XFAIL: darwin

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>

char buffer[10000];

__attribute__((noinline)) static void Symbolize() {
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
  for (char *p = buffer; strlen(p); p += strlen(p) + 1)
    printf("%s\n", p);
}

struct Symbolizer {
  __attribute__((noinline)) ~Symbolizer() { Symbolize(); }
};

// NODEMANGLE: in _ZN10SymbolizerD2Ev
// CHECK: in Symbolizer::~Symbolizer
int main() {
  Symbolizer();
  return 0;
}
