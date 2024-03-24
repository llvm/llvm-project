// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <malloc_np.h>
#include <cassert>
int main(int argc, char **argv) {
  const int flags = MALLOCX_ZERO;
  char *a = reinterpret_cast<char *>(mallocx(1024, 0));
  char index = a[1023];
  dallocx(a, 0);
  char *p = reinterpret_cast<char *>(mallocx(16, flags));
  p = reinterpret_cast<char *>(rallocx(p, 1024, flags));
  size_t asize = sallocx(p, flags);
  dallocx(p, flags);
  assert(asize >= 1024);
  return 0;
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*jemalloc.cpp:}}[[@LINE-2]]
}

