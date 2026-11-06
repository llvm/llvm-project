// RUN: %clangxx -O0 -g %s -o %t
// RUN: not %run %t src_zero 2>&1 | FileCheck %s --check-prefix=SRC-ZERO
// RUN: not %run %t src_small 2>&1 | FileCheck %s --check-prefix=SRC-SMALL
// RUN: not %run %t cat_src_zero 2>&1 | FileCheck %s --check-prefix=CAT-SRC-ZERO
// RUN: not %run %t cat_src_small 2>&1 | FileCheck %s --check-prefix=CAT-SRC-SMALL

// REQUIRES: asan || msan
// UNSUPPORTED: target={{.*linux.*}} && !glibc-2.38, target={{.*solaris.*}}

#include <string.h>

#include "sanitizer_common/sanitizer_specific.h"

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  alignas(8) char src[8] = "abcd";
  char dst[8] = {};
  make_mem_bad(src + 3, 5);

  if (strcmp(argv[1], "src_zero") == 0) {
    // SRC-ZERO: strlcpy
    // SRC-ZERO: SUMMARY: {{Address|Memory}}Sanitizer:
    return strlcpy(dst, src, 0) == 99;
  }
  if (strcmp(argv[1], "src_small") == 0) {
    // SRC-SMALL: strlcpy
    // SRC-SMALL: SUMMARY: {{Address|Memory}}Sanitizer:
    return strlcpy(dst, src, 2) == 99;
  }
  if (strcmp(argv[1], "cat_src_zero") == 0) {
    // CAT-SRC-ZERO: {{(#0 .* in (wrap_|___interceptor_)?|Uninitialized bytes in )}}strlcat
    // CAT-SRC-ZERO: SUMMARY: {{Address|Memory}}Sanitizer:
    return strlcat(dst, src, 0) == 99;
  }
  if (strcmp(argv[1], "cat_src_small") == 0) {
    // CAT-SRC-SMALL: {{(#0 .* in (wrap_|___interceptor_)?|Uninitialized bytes in )}}strlcat
    // CAT-SRC-SMALL: SUMMARY: {{Address|Memory}}Sanitizer:
    return strlcat(dst, src, 2) == 99;
  }
  return 1;
}
