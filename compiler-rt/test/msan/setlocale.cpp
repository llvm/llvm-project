// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <locale.h>
#include <stdlib.h>

int main(void) {
  char *locale = setlocale (LC_ALL, "");
  assert(locale);
  if (locale[0])
    exit(0);
  return 0;
}
