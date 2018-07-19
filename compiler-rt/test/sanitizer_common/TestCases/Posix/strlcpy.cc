// RUN: %clangxx -O0 -g %s -o %t && %run %t

// UNSUPPORTED: linux

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  const char src[] = "abc";
  char dst[7] = {'x', 'y', 'z', 0};
  size_t len;

  len = strlcpy(dst, src, sizeof(dst));
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

int main(void) {
  test1();
  test2();
  test3();
  test4();

  // CHECK: abc 3 abc 3 xyz 3  0

  return 0;
}
