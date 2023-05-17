// RUN: %clangxx -O0 -g %s -o %t -lutil && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <libutil.h>

int main(void) {
  printf("hexdump");
  char *line;
  size_t lineno = 0, len;
  const char *delim = "\\\\#";
  FILE *fp = fopen("/etc/fstab", "r");
  assert(fp);
  line = fparseln(fp, &len, &lineno, delim, 0);
  hexdump(line, len, nullptr, 0);
  free(line);
  fclose(fp);
  assert(lineno != 0);
  assert(len > 0);

  return 0;
}
