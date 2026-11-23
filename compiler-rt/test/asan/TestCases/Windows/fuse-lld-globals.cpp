// REQUIRES: lld-available

// RUN: %clangxx_asan -fuse-ld=lld -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <string.h>
int main(int argc, char **argv) {
  static char XXX[10];
  static char YYY[10];
  static char ZZZ[10];
  memset(XXX, 0, 10);
  memset(YYY, 0, 10);
  memset(ZZZ, 0, 10);
  int res = YYY[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*fuse-lld-globals.cpp:}}[[@LINE-2]]
  // CHECK: {{0x.* is located 0 bytes after global variable}}
  // CHECK:   {{.*YYY.* of size 10}}
  res += XXX[argc] + ZZZ[argc];
  return res;
}
