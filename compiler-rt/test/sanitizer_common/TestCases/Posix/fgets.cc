// RUN: %clangxx -g %s -o %t && %run %t

#include <stdio.h>

int main(int argc, char **argv) {
  FILE *fp;
  char buf[2];
  char *s;

  fp = fopen(argv[0], "r");
  if (!fp)
    return 1;

  s = fgets(buf, sizeof(buf), fp);
  if (!s)
    return 2;

  fclose(fp);
  return 0;
}
