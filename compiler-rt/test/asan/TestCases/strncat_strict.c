// Test strict_string_checks option in strncat function
// RUN: %clang_asan %s -o %t
// RUN: not %run %t test1 2>&1 | FileCheck %s --check-prefix=CHECK1-NONSTRICT --check-prefix=CHECK1
// RUN: %env_asan_opts=strict_string_checks=false not  %run %t test1 2>&1 | FileCheck %s --check-prefix=CHECK1-NONSTRICT --check-prefix=CHECK1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t test1 2>&1 | FileCheck %s --check-prefix=CHECK1-STRICT --check-prefix=CHECK1
// RUN: not %run %t test2 2>&1 | FileCheck %s --check-prefix=CHECK2-NONSTRICT --check-prefix=CHECK2
// RUN: %env_asan_opts=strict_string_checks=false not  %run %t test2 2>&1 | FileCheck %s --check-prefix=CHECK2-NONSTRICT --check-prefix=CHECK2
// RUN: %env_asan_opts=strict_string_checks=true not %run %t test2 2>&1 | FileCheck %s --check-prefix=CHECK2-STRICT --check-prefix=CHECK2

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void test1(char *to, int to_size, char *from) {
  // One of arguments points to not allocated memory.
  char* r = strncat(to + to_size, from, 2);
}

void test2(char *to, int to_size, char *from) {
  // "to" is not zero-terminated.
  memset(to, 'z', to_size);
  char* r = strncat(to, from, 1);
}

int main(int argc, char **argv) {
  size_t to_size = 100;
  char *to = (char*)malloc(to_size);
  size_t from_size = 20;
  char *from = (char*)malloc(from_size);
  memset(from, 'z', from_size);
  from[from_size - 1] = '\0';
  if (argc != 2) return 1;
  if (!strcmp(argv[1], "test1")) test1(to, to_size, from);
  // CHECK1: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK1-STRICT: READ of size 1
  // CHECK1-NONSTRICT: WRITE of size 3
  if (!strcmp(argv[1], "test2")) test2(to, to_size, from);
  // CHECK2: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK2-STRICT: READ of size 101
  // CHECK2-NONSTRICT: WRITE of size 2
  free(to);
  free(from);
  return 0;
}
