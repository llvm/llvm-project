// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>

#if !defined(__GLIBC_PREREQ)
#define __GLIBC_PREREQ(a, b) 0
#endif

// getauxval() used instead of sysconf() in GetPageSize() is defined starting
// glbc version 2.16.
#if __GLIBC_PREREQ(2, 16)
extern "C" long sysconf(int name) {
  fprintf(stderr, "sysconf wrapper called\n");
  return 0;
}
#endif  // defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 16)

int main() {
  // All we need to check is that the sysconf() interceptor defined above was
  // not called. Should it get called, it will crash right there, any
  // instrumented code executed before sanitizer init is finished will crash
  // accessing non-initialized sanitizer internals. Even if it will not crash
  // in some configuration, it should never be called anyway.
  fprintf(stderr, "Passed\n");
  // CHECK-NOT: sysconf wrapper called
  // CHECK: Passed
  // CHECK-NOT: sysconf wrapper called
  return 0;
}
