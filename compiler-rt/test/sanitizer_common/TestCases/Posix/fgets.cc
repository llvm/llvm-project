// RUN: %clangxx -g %s -o %t && %run %t

#include <stdio.h>

int main(void) {
  FILE *fp;
  char buf[2];
  char *s;

  fp = fopen("/etc/passwd", "r");
  if (!fp)
    return 1;

  s = fgets(buf, sizeof(buf), fp);
  if (!s)
    return 2;

  fclose(fp);
  return 0;
}
