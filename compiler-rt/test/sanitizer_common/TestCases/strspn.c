// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  size_t r;
  char s1[] = "ab";
  char s2[] = "ac";
  r = strspn(s1, s2);
  assert(r == 1);
  return 0;
}
