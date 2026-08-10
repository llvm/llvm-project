// Test strict_str`ing_checks option in strspn function
// RUN: %clang_asan %s -o %t && %run %t 2>&1
// RUN: %env_asan_opts=strict_string_checks=false %run %t 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  size_t size = 100;
  char fill = 'o';
  char *s1 = (char*)malloc(size);
  char *s2 = (char*)malloc(2);
  memset(s1, fill, size);
  s1[0] = s2[0] = 'z';
  s2[1] = '\0';
  size_t r = strspn(s1, s2);
  // CHECK: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK: READ of size 101
  assert(r == 1);
  free(s1);
  free(s2);
  return 0;
}
