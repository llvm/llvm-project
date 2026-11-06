// RUN: %clangxx -O0 -g %s -o %t
// RUN: %run %t
// RUN: %env_tool_opts=strict_string_checks=1 %run %t

// UNSUPPORTED: target={{.*linux.*}} && !glibc-2.38

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sanitizer_common/sanitizer_specific.h"

void test1() {
  const char src[] = "abc";
  char dst[7] = {'x', 'y', 'z', 0};
  size_t len;

  len = strlcat(dst, src, sizeof(dst));
  printf("%s %zu ", dst, len);
}

void test2() {
  const char src[] = "abc";
  char dst[7] = {0};
  size_t len;

  len = strlcat(dst, src, sizeof(dst));
  printf("%s %zu ", dst, len);
}

void test3() {
  const char src[] = "abc";
  char dst[4] = {'x', 'y', 'z', 0};
  size_t len;

  len = strlcat(dst, src, sizeof(dst));
  printf("%s %zu ", dst, len);
}

void test4() {
  const char src[] = "";
  char dst[4] = {'x', 'y', 'z', 0};
  size_t len;

  len = strlcat(dst, src, sizeof(dst));
  printf("%s %zu\n", dst, len);
}

void test_zero_size() {
  alignas(16) char dst[16];
  make_mem_bad(dst, sizeof(dst));
  assert(strlcat(dst, "abc", 0) == 3);
  make_mem_good(dst, sizeof(dst));

  char *empty = (char *)malloc(0);
  assert(strlcat(empty, "abc", 0) == 3);
  free(empty);
}

void test_unterminated_dst() {
  char *dst = (char *)malloc(4);
  memcpy(dst, "abcd", 4);
  assert(strlcat(dst, "ef", 4) == 6);
  assert(memcmp(dst, "abcd", 4) == 0);
  free(dst);
}

int main(void) {
  test1();
  test2();
  test3();
  test4();
  test_zero_size();
  test_unterminated_dst();

  // CHECK: xyzabc 6 abc 3 xyz 3 xyz 3

  return 0;
}
