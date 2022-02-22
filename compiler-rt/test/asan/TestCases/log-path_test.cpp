// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// UNSUPPORTED: ios
//
// The for loop in the backticks below requires bash.
// REQUIRES: shell
//
// RUN: %clangxx_asan  %s -o %t

// Regular run.
// RUN: not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.out

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %env_asan_opts=log_path=%t.log not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.log.*

// Invalid log_path in existing directory.
// RUN: %env_asan_opts=log_path=/INVALID not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-INVALID < %t.out

// Directory of log_path can't be created.
// RUN: %env_asan_opts=log_path=/dev/null/INVALID not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-BAD-DIR < %t.out

// Too long log_path.
// RUN: %env_asan_opts=log_path=`for((i=0;i<10000;i++)); do echo -n $i; done` \
// RUN:   not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-LONG < %t.out

// Run w/o errors should not produce any log.
// RUN: rm -f %t.log.*
// RUN: %env_asan_opts=log_path=%t.log  %run %t ARG ARG ARG
// RUN: not cat %t.log.*

// FIXME: log_path is not supported on Windows yet.
// XFAIL: windows-msvc

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
// CHECK-INVALID: ERROR: Can't open file: /INVALID
// CHECK-BAD-DIR: ERROR: Can't create directory: /dev/null
// CHECK-LONG: ERROR: Path is too long: 01234
