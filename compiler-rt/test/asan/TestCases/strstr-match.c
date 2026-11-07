// Test which part of the haystack strstr and strcasestr check when they find a
// match: everything up to the end of the match, and nothing after it.

// RUN: %clang_asan %s -o %t
// RUN: %run %t after 2>&1 | FileCheck %s --check-prefix=OK
// RUN: not %run %t inside 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not %run %t before 2>&1 | FileCheck %s --check-prefix=BAD
// Disable other interceptors because strlen may be called inside strstr.
// RUN: %env_asan_opts=intercept_strstr=false:replace_str=false %run %t inside 2>&1 | FileCheck %s --check-prefix=OK
// With strict_string_checks the whole haystack is checked.
// RUN: %env_asan_opts=strict_string_checks=true not %run %t after 2>&1 | FileCheck %s --check-prefix=STRICT

// Windows has no strcasestr.
// XFAIL: target={{.*windows-(msvc.*|gnu)}}
// On Android and FreeBSD strstr calls memchr to find the terminator of the
// haystack, and the interceptor of memchr reports the poison after the match.
// UNSUPPORTED: android, target={{.*freebsd.*}}

#define _GNU_SOURCE
#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  char *haystack = (char *)malloc(64);
  memset(haystack, 'x', 63);
  haystack[63] = 0;
  memcpy(haystack + 16, "needle", 6);
  // One poisoned granule: after, inside, or in front of the match in [16, 22).
  size_t pos = !strcmp(argv[1], "after")    ? 24
               : !strcmp(argv[1], "inside") ? 16
                                            : 8;
  __asan_poison_memory_region(haystack + pos, 8);
  char *r1 = strstr(haystack, "needle");
  char *r2 = strcasestr(haystack, "NEEDLE");
  // BAD: ERROR: AddressSanitizer: use-after-poison
  // BAD: READ of size 22
  // BAD: #{{[01]}} {{.*}} in {{.*}}strstr
  // STRICT: ERROR: AddressSanitizer: use-after-poison
  // STRICT: READ of size 64
  // STRICT: #{{[01]}} {{.*}} in {{.*}}strstr
  assert(r1 == haystack + 16);
  assert(r2 == haystack + 16);
  __asan_unpoison_memory_region(haystack + pos, 8);
  free(haystack);
  // OK: done
  // OK-NOT: AddressSanitizer
  fprintf(stderr, "done\n");
  return 0;
}
