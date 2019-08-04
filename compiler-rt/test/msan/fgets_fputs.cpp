// RUN: %clangxx_msan -g %s -o %t
// RUN: %run %t
// RUN: not %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-FPUTS
// RUN: not %run %t 3 3 2>&1 | FileCheck %s --check-prefix=CHECK-PUTS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_fgets() {
  FILE *fp = fopen("/dev/zero", "r");
  char c;

  if (!fgets(&c, 1, fp))
    return 1;

  if (c == '1') // No error
    return 2;

  fclose(fp);
  return 0;
}

int test_fputs() {
  FILE *fp = fopen("/dev/null", "w");
  char buf[2];
  fputs(buf, fp); // BOOM
  return fclose(fp);
}

void test_puts() {
  char buf[2];
  puts(buf); // BOOM
}

int main(int argc, char *argv[]) {
  if (argc == 1)
    test_fgets();
  else if (argc == 2)
    test_fputs();
  else
    test_puts();
  return 0;
}

// CHECK-FPUTS: Uninitialized bytes in __interceptor_fputs at offset 0 inside
// CHECK-PUTS: Uninitialized bytes in __interceptor_puts at offset 0 inside
