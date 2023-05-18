// Test strict_string_checks option in strchr function
// RUN: %clang_asan %s -o %t && %run %t 2>&1
// RUN: %env_asan_opts=strict_string_checks=false %run %t 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  size_t size = 100;
  char fill = 'o';
  char *s = (char*)malloc(size);
  memset(s, fill, size);
  char c = 'o';
  char* r = strchr(s, c);
  // CHECK: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK: READ of size 101
  assert(r == s);
  free(s);
  return 0;
}
