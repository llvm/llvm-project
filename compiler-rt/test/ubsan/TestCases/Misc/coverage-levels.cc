// Test various levels of coverage
//
// FIXME: Port the environment variable logic below for the lit shell.
// REQUIRES: shell
//
// RUN: rm -rf %t-dir && mkdir %t-dir
// RUN: %clangxx -fsanitize=shift                        -DGOOD_SHIFT=1 -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN
// RUN: %clangxx -fsanitize=undefined                    -DGOOD_SHIFT=1 -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN

// Also works without any sanitizer.
// RUN: %clangxx                                         -DGOOD_SHIFT=1 -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN

// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_WARN
// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=bb  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2 --check-prefix=CHECK_WARN
// RUN: %clangxx -fsanitize=shift -O1 -fsanitize-coverage=edge  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:verbosity=1:coverage_dir='"%t-dir"' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3 --check-prefix=CHECK_WARN

// Coverage is not yet implemented in TSan.
// XFAIL: ubsan-tsan
// UNSUPPORTED: ubsan-standalone-static
// No coverage support
// UNSUPPORTED: openbsd

volatile int sink;
int main(int argc, char **argv) {
  int shift = argc * 32;
#if GOOD_SHIFT
  shift = 3;
#endif
  if ((argc << shift) == 16)  // False.
    return 1;
  return 0;
}

// CHECK_WARN: shift exponent 32 is too large
// CHECK_NOWARN-NOT: ERROR
// FIXME: Currently, coverage instrumentation kicks in after ubsan, so we get
// more than the minimal number of instrumented blocks.
// FIXME: Currently, ubsan with -fno-sanitize-recover and w/o asan will fail
// to dump coverage.
// CHECK1:  1 PCs written
// CHECK2:  2 PCs written
// CHECK3:  2 PCs written
