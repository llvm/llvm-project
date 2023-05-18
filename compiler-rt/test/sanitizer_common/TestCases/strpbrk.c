// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <string.h>


int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "ad";
  char s2[] = "cd";
  r = strpbrk(s1, s2);
  assert(r == s1 + 1);
  return 0;
}
