// RUN: %clangxx -O0 -g %s -o %t && %run %t
// UNSUPPORTED: linux

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  FILE *fp;
  size_t len;
  char *s;

  fp = fopen("/etc/hosts", "r");
  if (!fp)
    exit(1);

  s = fgetln(fp, &len);

  printf("%.*s\n", (int)len, s);

  if (fclose(fp) == EOF)
    exit(1);

  return 0;
}
