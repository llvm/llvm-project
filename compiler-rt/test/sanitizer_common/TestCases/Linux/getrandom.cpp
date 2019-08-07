// RUN: %clangxx -O2 %s -o %t && %run %t
// UNSUPPORTED: android
//

#include <sys/types.h>

#if !defined(__GLIBC_PREREQ)
#define __GLIBC_PREREQ(a, b) 0
#endif

#if __GLIBC_PREREQ(2, 25)
#include <sys/random.h>
#endif

int main() {
  char buf[16];
  ssize_t n = 1;
#if __GLIBC_PREREQ(2, 25)
  n = getrandom(buf, sizeof(buf), 0);
#endif
  return (int)(n <= 0);
}
