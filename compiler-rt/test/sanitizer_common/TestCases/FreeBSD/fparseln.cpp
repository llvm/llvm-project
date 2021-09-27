// RUN: %clangxx -O0 -g %s -o %t -lutil && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <libutil.h>

int main(void) {
  printf("fparseln\n");

  FILE *fp = fopen("/etc/fstab", "r");
  assert(fp);

  int flags = FPARSELN_UNESCALL;
  const char *delim = "\\\\#";
  size_t lineno = 0, len;
  char *line;
  while ((line = fparseln(fp, &len, &lineno, delim, flags))) {
    printf("lineno: %zu, length: %zu, line: %s\n", lineno, len, line);
    free(line);
  }

  // CHECK: fparseln

  return 0;
}
