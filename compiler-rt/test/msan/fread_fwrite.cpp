// RUN: %clangxx_msan -g %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %t 1

#include <stdio.h>
#include <stdlib.h>

int test_fread() {
  FILE *f = fopen("/dev/zero", "r");
  char c;
  unsigned read = fread(&c, sizeof(c), 1, f);
  fclose(f);
  if (c == '1') // No error
    return 1;
  return 0;
}

int test_fwrite() {
  FILE *f = fopen("/dev/null", "w");
  char c;
  if (fwrite(&c, sizeof(c), 1, f) != sizeof(c)) // BOOM
    return 1;
  return fclose(f);
}

int main(int argc, char *argv[]) {
  if (argc > 1)
    test_fread();
  else
    test_fwrite();
  return 0;
}

// CHECK: Uninitialized bytes in fwrite at offset 0 inside
