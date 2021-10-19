// RUN: %clang -O2 %s -o %t && %run %t
// UNSUPPORTED: android, netbsd, darwin, solaris
//

#include <sys/types.h>
#include <errno.h>

#if !defined(__GLIBC_PREREQ)
#define __GLIBC_PREREQ(a, b) 0
#endif

#if (defined(__linux__) && __GLIBC_PREREQ(2, 25)) || defined(__FreeBSD__)
#define HAS_GETRANDOM
#endif

#if defined(HAS_GETRANDOM)
#include <sys/random.h>
#endif

int main() {
  char buf[16];
  ssize_t n = 1;
#if defined(HAS_GETRANDOM)
  n = getrandom(buf, sizeof(buf), 0);
  if (n == -1 && errno == ENOSYS)
    n = 1;
#endif
  return (int)(n <= 0);
}
