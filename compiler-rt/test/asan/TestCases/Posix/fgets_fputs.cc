// RUN: %clangxx_asan -g %s -o %t
// RUN: echo data > %t-testdata
// RUN: not %run %t 1 %t-testdata 2>&1 | FileCheck %s --check-prefix=CHECK-FGETS
// RUN: not %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-FPUTS
// RUN: not %run %t 3 2>&1 | FileCheck %s --check-prefix=CHECK-PUTS
// XFAIL: android

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_fgets(const char *testfile) {
  char buf[2];
  FILE *fp = fopen(testfile, "r");
  assert(fp);
  fgets(buf, sizeof(buf) + 1, fp); // BOOM
  fclose(fp);
  return 0;
}

int test_fputs() {
  char buf[1] = {'x'}; // Note: not nul-terminated
  FILE *fp = fopen("/dev/null", "w");
  assert(fp);
  fputs(buf, fp); // BOOM
  fclose(fp);
  return 0;
}

int test_puts() {
  char *p = strdup("x");
  free(p);
  puts(p); // BOOM
  return 0;
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);
  int testno = argv[1][0] - '0';
  if (testno == 1) {
    assert(argc == 3);
    return test_fgets(argv[2]);
  }
  if (testno == 2)
    return test_fputs();
  if (testno == 3)
    return test_puts();
  return 1;
}

// CHECK-FGETS: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FGETS: #{{.*}} in {{(wrap_|__interceptor_)?}}fgets
// CHECK-FPUTS: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FPUTS: #{{.*}} in {{(wrap_|__interceptor_)?}}fputs
// CHECK-PUTS: {{.*ERROR: AddressSanitizer: heap-use-after-free}}
// CHECK-PUTS: #{{.*}} in {{(wrap_|__interceptor_)?}}puts
