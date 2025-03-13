// REQUIRES: static-libs
// REQUIRES: have_disable_symbolizer_path_search

// RUN: %clangxx %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

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

// CHECK: Symbolizer path search is disabled in the runtime build configuration"
