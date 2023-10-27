// REQUIRES: (linux && !android) || freebsd

// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: touch %t-dir/a %t-dir/b %t-dir/c

// RUN: %clang %s -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  struct dirent **dirpp = NULL;
  int count = scandir(TEMP_DIR, &dirpp, NULL, NULL);
  fprintf(stderr, "count is %d\n", count);
  if (count >= 0) {
    for (int i = 0; i < count; ++i) {
      fprintf(stderr, "found %s\n", dirpp[i]->d_name);
      free(dirpp[i]);
    }
    free(dirpp);
  }
  return 0;
}
