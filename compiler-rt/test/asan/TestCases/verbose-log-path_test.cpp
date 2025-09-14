// UNSUPPORTED: system-windows

// RUN: rm -rf %t-dir && mkdir -p %t-dir
// RUN: %clangxx_asan %s -o %t-dir/verbose-log-path_test-binary

// Good log_path.
// RUN: rm -f %t-dir/asan.log.*
// RUN: %env_asan_opts=log_path=%t-dir/asan.log:log_exe_name=1 not %run %t-dir/verbose-log-path_test-binary 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t-dir/asan.log.verbose-log-path_test-binary.*

// FIXME: only FreeBSD, NetBSD and Linux have verbose log paths now.
// XFAIL: target={{.*windows-msvc.*}},android
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
