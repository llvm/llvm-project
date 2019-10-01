// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// REQUIRES: !android

#include <assert.h>
#include <strings.h>
#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  char buf[100];
  assert(0 == __msan_test_shadow(buf, sizeof(buf)));
  // *& to suppress bzero-to-memset optimization.
  (*&bzero)(buf, 50);
  assert(50 == __msan_test_shadow(buf, sizeof(buf)));
  return 0;
}
