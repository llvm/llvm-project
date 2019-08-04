// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  void *p = dlopen("/bad/file/name", RTLD_NOW);
  assert(!p);
  char *s = dlerror();
  printf("%s, %zu\n", s, strlen(s));
  return 0;
}
