// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

volatile int *p = 0;

int main() {
  {
    int x = 0;
    p = &x;
  }
  *p = 5;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope.cpp:[[@LINE-2]]
  // CHECK: Address 0x{{.*}} is located in stack of thread T{{.*}} at offset [[OFFSET:[^ ]+]] in frame
  // {{\[}}[[OFFSET]], {{[0-9]+}}) 'x'
  return 0;
}
