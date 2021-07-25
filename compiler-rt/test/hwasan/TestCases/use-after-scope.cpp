// RUN: %clangxx_asan -O1 -fsanitize-address-use-after-scope %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// -fsanitize-address-use-after-scope is now on by default:
// RUN: %clangxx_asan -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
//
// Not expected to work yet with HWAsan.
// XFAIL: *

volatile int *p = 0;

int main() {
  {
    int x = 0;
    p = &x;
  }
  *p = 5; // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope.cpp:[[@LINE-2]]
  // CHECK: Address 0x{{.*}} is located in stack of thread T{{.*}} at offset [[OFFSET:[^ ]+]] in frame
  // {{\[}}[[OFFSET]], {{[0-9]+}}) 'x'
  return 0;
}
