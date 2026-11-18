// REQUIRES: powerpc64-target-arch

// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=external_symbolizer_path=%p/Inputs/sanitizer_default_arch/llvm-symbolizer \
 // RUN:   %run %t 2>&1 | FileCheck %s

#include <sanitizer/common_interface_defs.h>

static void Symbolize() {
  char buffer[100];
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
}

int main() {
  // CHECK: "--default-arch=powerpc64"
  Symbolize();
}
