// XFAIL: msvc
// RUN: %clangxx_asan -O0 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s %p/Helpers/underflow.cpp -o %t && not %run %t 2>&1 | FileCheck %s

// aix puts XXX and YYY at very different addresses. For example YYY is 0x20004340, XXX is 0x20000c20
// This address allocation does not match the assumption in https://reviews.llvm.org/D38056.
// It was awared that this case may be not reliable on other OS.
// UNSUPPORTED: target={{.*aix.*}}

int XXX[2] = {2, 3};
extern int YYY[];
#include <string.h>
int main(int argc, char **argv) {
  memset(XXX, 0, 2*sizeof(int));
  // CHECK: {{READ of size 4 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*global-underflow.cpp:}}[[@LINE+3]]
  // CHECK: {{0x.* is located 4 bytes before global variable}}
  // CHECK:   {{.*YYY.* of size 12}}
  int res = YYY[-1];
  return res;
}
