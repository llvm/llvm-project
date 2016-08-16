// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t

// XFAIL: target-is-mips64el

#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  unsigned char s[L_ctermid + 1];
  char *res = ctermid((char *)s);
  if (res)
    printf("%zd\n", strlen(res));
  return 0;
}
