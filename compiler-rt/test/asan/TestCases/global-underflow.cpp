// RUN: %clangxx_asan -O0 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s

int XXX[2] = {2, 3};
extern int YYY[];
#include <string.h>
int main(int argc, char **argv) {
  memset(XXX, 0, 2*sizeof(int));
  // CHECK: {{READ of size 4 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*global-underflow.cpp:}}[[@LINE+3]]
  // CHECK: {{0x.* is located 4 bytes to the left of global variable}}
  // CHECK:   {{.*YYY.* of size 12}}
  int res = YYY[-1];
  return res;
}
