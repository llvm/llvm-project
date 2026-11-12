// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  const char *p = strsignal(SIGSEGV);
  assert(p);
  printf("%s %zu\n", p, strlen(p));
  return 0;
}
