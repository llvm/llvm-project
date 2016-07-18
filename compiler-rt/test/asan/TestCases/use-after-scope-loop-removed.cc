// RUN: %clangxx_asan -O1 -fsanitize-address-use-after-scope %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
//
// FIXME: Compiler removes for-loop but keeps x variable. For unknown reason
// @llvm.lifetime.* are not emitted for x.
// XFAIL: *

#include <stdlib.h>

int *p;

int main() {
  for (int i = 0; i < 3; i++) {
    int x;
    p = &x;
  }
  return **p;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
}
