// RUN: %clangxx_asan %s -o %T/verbose-log-path_test-binary

// The glob below requires bash.
// REQUIRES: shell

// Good log_path.
// RUN: rm -f %T/asan.log.*
// RUN: %env_asan_opts=log_path=%T/asan.log:log_exe_name=1 not %run %T/verbose-log-path_test-binary 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %T/asan.log.verbose-log-path_test-binary.*

// FIXME: only FreeBSD and Linux have verbose log paths now.
// XFAIL: win32,android
// UNSUPPORTED: ios

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc > 2) return 0;
  char *x = (char*)malloc(10);
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  free(x);
  return res;
}
// CHECK-ERROR: ERROR: AddressSanitizer
