// RUN: %clang_dfsan %s -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -mllvm -dfsan-debug-nonzero-labels -o %t && DFSAN_OPTIONS=warn_unimplemented=1 %run %t 2>&1 | FileCheck %s
// RUN: %clang_dfsan %s -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -mllvm -dfsan-debug-nonzero-labels -o %t && DFSAN_OPTIONS=warn_unimplemented=0 %run %t 2>&1 | count 0
// RUN: %clang_dfsan %s -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -mllvm -dfsan-debug-nonzero-labels -o %t && DFSAN_OPTIONS=warn_nonzero_labels=1 %run %t 2>&1 | FileCheck --check-prefix=CHECK-NONZERO %s
//
// REQUIRES: x86_64-target-arch

// Tests that flags work correctly.

#include <sanitizer/dfsan_interface.h>

int f(int i) {
  return i;
}

int main(void) {
  int i = 1;
  dfsan_label i_label = 2;
  dfsan_set_label(i_label, &i, sizeof(i));

  // CHECK: WARNING: DataFlowSanitizer: call to uninstrumented function f
  // CHECK-NOT: WARNING: DataFlowSanitizer: saw nonzero label
  // CHECK-NONZERO: WARNING: DataFlowSanitizer: saw nonzero label
  f(i);

  return 0;
}
