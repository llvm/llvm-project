// RUN: %clangxx -O0 -g %s -o %t
// RUN: not %run %t cpy 2>&1 | FileCheck %s --check-prefix=CPY
// RUN: not %run %t cat 2>&1 | FileCheck %s --check-prefix=CAT

// REQUIRES: asan
// UNSUPPORTED: target={{.*linux.*}} && !glibc-2.38, target={{.*solaris.*}}

#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  if (strcmp(argv[1], "cpy") == 0) {
    volatile size_t alloc_size = 3;
    char *dst = (char *)malloc(alloc_size);
    // CPY: ERROR: AddressSanitizer: heap-buffer-overflow
    // CPY: WRITE of size 4
    // CPY: #0 {{.*}} in {{(wrap_|___interceptor_)?}}strlcpy
    size_t res = strlcpy(dst, "abc", 4);
    free(dst);
    return res == 99;
  }

  if (strcmp(argv[1], "cat") == 0) {
    volatile size_t alloc_size = 4;
    char *dst = (char *)malloc(alloc_size);
    memcpy(dst, "ab", 3);
    // CAT: ERROR: AddressSanitizer: heap-buffer-overflow
    // CAT: WRITE of size 3
    // CAT: #0 {{.*}} in {{(wrap_|___interceptor_)?}}strlcat
    size_t res = strlcat(dst, "cd", 5);
    free(dst);
    return res == 99;
  }

  return 1;
}
