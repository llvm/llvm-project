// RUN: %clangxx -g %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv) {
  FILE *fp = fopen(argv[0], "r");
  assert(fp);

  // the file should be at least one character long, always
  assert(fgetc(fp) != EOF);
  // POSIX guarantees being able to ungetc() at least one character
  assert(ungetc('X', fp) != EOF);
  // check whether ungetc() worked
  assert(getc(fp) == 'X');

  assert(!fclose(fp));
  return 0;
}
