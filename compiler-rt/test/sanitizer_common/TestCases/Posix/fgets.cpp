// RUN: %clangxx -g %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv) {
  FILE *fp = fopen(argv[0], "r");
  assert(fp);

  char buf[2];
  char *s = fgets(buf, sizeof(buf), fp);
  assert(s);

  assert(!fclose(fp));
  return 0;
}
