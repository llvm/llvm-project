// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t
// RUN: %run %t
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O3 %s -o %t
// RUN: %run %t

#include <assert.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  char s[20] = "string";
  __msan_poison(s, sizeof(s));
  __msan_unpoison_string(s);
  assert(__msan_test_shadow(s, sizeof(s)) == strlen("string") + 1);
  return 0;
}
