// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=invalid_option_name=10,verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-V1
// RUN: %env_tool_opts=invalid_option_name=10 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-V0

#include <stdio.h>

int main() {
  fprintf(stderr, "done\n");
}

// CHECK-V1: WARNING: found 1 unrecognized
// CHECK-V1:     invalid_option_name
// CHECK-V0-NOT: WARNING: found 1 unrecognized
// CHECK-V0-NOT:     invalid_option_name
// CHECK: done
