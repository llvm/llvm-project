// RUN: %clang  %s -o %t && %run %t
// XFAIL: ubsan,lsan

#include <assert.h>
#include <sys/mman.h>

int main() {
  assert(0 == mlockall(MCL_CURRENT));
  assert(0 == mlock((void *)0x12345, 0x5678));
  assert(0 == munlockall());
  assert(0 == munlock((void *)0x987, 0x654));
}

