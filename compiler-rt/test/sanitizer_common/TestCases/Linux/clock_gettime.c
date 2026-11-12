// RUN: %clang %s -Wl,-as-needed -o %t && %run %t
// Regression test for PR15823
// (http://llvm.org/bugs/show_bug.cgi?id=15823).
#include <stdio.h>
#include <time.h>

int main() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return 0;
}
