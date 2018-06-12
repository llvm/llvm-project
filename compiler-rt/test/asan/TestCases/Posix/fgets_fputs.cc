// RUN: %clangxx_asan -g %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-FGETS
// RUN: not %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-FPUTS
// RUN: not %run %t 3 3 2>&1 | FileCheck %s --check-prefix=CHECK-PUTS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_fgets() {
  FILE *fp = fopen("/etc/passwd", "r");
  char buf[2];
  fgets(buf, sizeof(buf) + 1, fp); // BOOM
  fclose(fp);
  return 0;
}

int test_fputs() {
  FILE *fp = fopen("/dev/null", "w");
  char buf[1] = {'x'}; // Note: not nul-terminated
  fputs(buf, fp);      // BOOM
  return fclose(fp);
}

void test_puts() {
  char *p = strdup("x");
  free(p);
  puts(p); // BOOM
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

// CHECK-FGETS: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FGETS: #{{.*}} in {{(wrap_|__interceptor_)?}}fgets
// CHECK-FPUTS: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FPUTS: #{{.*}} in {{(wrap_|__interceptor_)?}}fputs
// CHECK-PUTS: {{.*ERROR: AddressSanitizer: heap-use-after-free}}
// CHECK-PUTS: #{{.*}} in {{(wrap_|__interceptor_)?}}puts
