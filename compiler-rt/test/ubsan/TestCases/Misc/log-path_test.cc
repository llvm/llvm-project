// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android

// The globs below do not work in the lit shell.
// REQUIRES: shell

// RUN: %clangxx -fsanitize=undefined %s -O1 -o %t

// Regular run.
// RUN: %run %t -4 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.out

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %device_rm -f '%t.log.*'
// RUN: %env_ubsan_opts=log_path='"%t.log"' %run %t -4 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.log.*

// Run w/o errors should not produce any log.
// RUN: rm -f %t.log.*
// RUN: %device_rm -f '%t.log.*'
// RUN: %env_ubsan_opts=log_path='"%t.log"'  %run %t 4
// RUN: not cat %t.log.*

// FIXME: log_path is not supported on Windows yet.
// XFAIL: windows-msvc

#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
  double a = atof(argv[1]);
  unsigned int ai = (unsigned int) a;
  printf("%f %u\n", a, ai);
  return 0;
}

// CHECK-ERROR: runtime error: -4 is outside the range of representable values of type 'unsigned int'

