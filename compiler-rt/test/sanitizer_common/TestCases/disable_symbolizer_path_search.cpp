// REQUIRES: disable_symbolizer_path_search

// RUN: %clangxx %s -o %t
// RUN: %env_unset_tool_symbolizer_path \
// RUN: %env_tool_opts=verbosity=3 %run %t 2>&1 | FileCheck %s

// CHECK: Symbolizer path search is disabled in the runtime build configuration

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

static void Symbolize() {
  char buffer[100];
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
  printf("%s\n", buffer);
}

int main() { Symbolize(); }
