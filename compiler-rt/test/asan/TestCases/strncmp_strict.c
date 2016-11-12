// Test strict_string_checks option in strncmp function
// RUN: %clang_asan %s -o %t

// RUN: %env_asan_opts=strict_string_checks=false %run %t a 2>&1
// RUN: %env_asan_opts=strict_string_checks=true %run %t a 2>&1
// RUN: not %run %t b 2>&1 | FileCheck %s
// RUN: not %run %t c 2>&1 | FileCheck %s
// RUN: not %run %t d 2>&1 | FileCheck %s
// RUN: not %run %t e 2>&1 | FileCheck %s
// RUN: not %run %t f 2>&1 | FileCheck %s
// RUN: not %run %t g 2>&1 | FileCheck %s
// RUN: %env_asan_opts=strict_string_checks=false %run %t h 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t h 2>&1 | FileCheck %s
// RUN: %env_asan_opts=strict_string_checks=false %run %t i 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t i 2>&1 | FileCheck %s

// RUN: %env_asan_opts=strict_string_checks=false %run %t A 2>&1
// RUN: %env_asan_opts=strict_string_checks=true %run %t A 2>&1
// RUN: not %run %t B 2>&1 | FileCheck %s
// RUN: not %run %t C 2>&1 | FileCheck %s
// RUN: not %run %t D 2>&1 | FileCheck %s
// RUN: not %run %t E 2>&1 | FileCheck %s
// RUN: not %run %t F 2>&1 | FileCheck %s
// RUN: not %run %t G 2>&1 | FileCheck %s
// RUN: %env_asan_opts=strict_string_checks=false %run %t H 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t H 2>&1 | FileCheck %s
// RUN: %env_asan_opts=strict_string_checks=false %run %t I 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t I 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char **argv) {
  assert(argc >= 2);
  const size_t size = 100;
  char fill = 'o';
  char s1[size];
  char s2[size];
  memset(s1, fill, size);
  memset(s2, fill, size);

  int (*cmp_fn)(const char *, const char *, size_t);
  cmp_fn = islower(argv[1][0]) ? &strncmp : &strncasecmp;

  switch(tolower(argv[1][0])) {
    case 'a':
      s1[size - 1] = 'z';
      s2[size - 1] = 'x';
      for (int i = 0; i <= size; ++i)
        assert((cmp_fn(s1, s2, i) == 0) == (i < size));
      s1[size - 1] = '\0';
      s2[size - 1] = '\0';
      assert(cmp_fn(s1, s2, 2*size) == 0);
      break;
    case 'b':
      return cmp_fn(s1-1, s2, 1);
    case 'c':
      return cmp_fn(s1, s2-1, 1);
    case 'd':
      return cmp_fn(s1+size, s2, 1);
    case 'e':
      return cmp_fn(s1, s2+size, 1);
    case 'f':
      return cmp_fn(s1+1, s2, size);
    case 'g':
      return cmp_fn(s1, s2+1, size);
    case 'h':
      s1[size - 1] = '\0';
      assert(cmp_fn(s1, s2, 2*size) != 0);
      break;
    case 'i':
      s2[size - 1] = '\0';
      assert(cmp_fn(s1, s2, 2*size) != 0);
      break;
    // CHECK: {{.*}}ERROR: AddressSanitizer: stack-buffer-{{ov|und}}erflow on address
  }
  return 0;
}
