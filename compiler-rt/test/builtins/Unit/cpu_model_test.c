// REQUIRES: x86-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_x86

// FIXME: XFAIL the test because it is expected to return non-zero value.
// XFAIL: *
#include <stdio.h>

int main (void) {
#if defined(i386) || defined(__x86_64__)
  if(__builtin_cpu_supports("avx2"))
    return 4;
  else
    return 3;
#else
  printf("skipped\n");
  return 0;
#endif
}
