// RUN: %clangxx_asan -g %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-FWRITE
// RUN: not %run %t 1 2>&1 | FileCheck %s --check-prefix=CHECK-FREAD

#include <stdio.h>
#include <stdlib.h>

int test_fread() {
  FILE *f = fopen("/dev/zero", "r");
  char buf[2];
  fread(buf, sizeof(buf), 2, f); // BOOM
  fclose(f);
  return 0;
}

int test_fwrite() {
  FILE *f = fopen("/dev/null", "w");
  char buf[2];
  fwrite(buf, sizeof(buf), 2, f); // BOOM
  return fclose(f);
}

int main(int argc, char *argv[]) {
  if (argc > 1)
    test_fread();
  else
    test_fwrite();
  return 0;
}

// CHECK-FREAD: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FREAD: #{{.*}} in {{(wrap_|__interceptor_)?}}fread
// CHECK-FWRITE: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
// CHECK-FWRITE: #{{.*}} in {{(wrap_|__interceptor_)?}}fwrite
