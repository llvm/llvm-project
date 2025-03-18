// REQUIRES: disable_symbolizer_path_search

// RUN: %clangxx %s -o %t
// RUN: echo $ASAN_SYMBOLIZER_PATH
// RUN: export ASAN_SYMBOLIZER_PATH= TSAN_SYMBOLIZER_PATH= MSAN_SYMBOLIZER_PATH= UBSAN_SYMBOLIZER_PATH= 
// RUN: echo $ASAN_SYMBOLIZER_PATH
// RUN: %env_tool_opts=verbosity=3 %run %t 2>&1 | FileCheck %s

// CHECK: Symbolizer path search is disabled in the runtime build configuration

// REQUIRES: shell

// Mobile device will not have symbolizer in provided path.
// UNSUPPORTED: ios, android

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

static void Symbolize() {
  char buffer[100];
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
  printf("%s\n", buffer);
}

int main() {
  Symbolize();
}
