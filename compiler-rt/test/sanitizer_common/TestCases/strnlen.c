// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <string.h>
int main(int argc, char **argv) {
  const char *s = "mytest";
  assert(strnlen(s, 0) == 0UL);
  assert(strnlen(s, 1) == 1UL);
  assert(strnlen(s, 6) == strlen(s));
  assert(strnlen(s, 7) == strlen(s));
  return 0;
}
