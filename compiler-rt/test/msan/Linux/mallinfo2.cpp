// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// REQUIRES: glibc-2.33

#include <assert.h>
#include <malloc.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  struct mallinfo2 mi2 = mallinfo2();
  assert(__msan_test_shadow(&mi2, sizeof(mi2)) == -1);
  return 0;
}
