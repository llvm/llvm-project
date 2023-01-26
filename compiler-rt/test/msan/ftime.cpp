// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

// ftime() is deprecated on FreeBSD and NetBSD.
// UNSUPPORTED: target={{.*(freebsd|netbsd).*}}

#include <assert.h>
#include <sys/timeb.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  struct timeb tb;
  int res = ftime(&tb);
  assert(!res);
  assert(__msan_test_shadow(&tb, sizeof(tb)) == -1);
  return 0;
}
