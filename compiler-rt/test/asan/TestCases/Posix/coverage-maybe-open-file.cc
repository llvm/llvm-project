// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: rm -rf %T/coverage-maybe-open-file
// RUN: mkdir -p %T/coverage-maybe-open-file && cd %T/coverage-maybe-open-file
// RUN: %env_asan_opts=coverage=1 %run %t | FileCheck %s --check-prefix=CHECK-success
// RUN: %env_asan_opts=coverage=0 %run %t | FileCheck %s --check-prefix=CHECK-fail
// RUN: FileCheck %s <  test.sancov.packed -implicit-check-not={{.}} --check-prefix=CHECK-test
// RUN: cd .. && rm -rf %T/coverage-maybe-open-file

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <sanitizer/coverage_interface.h>

// FIXME: the code below might not work on Windows.
int main(int argc, char **argv) {
  int fd = __sanitizer_maybe_open_cov_file("test");
  if (fd > 0) {
    printf("SUCCESS\n");
    const char s[] = "test\n";
    write(fd, s, strlen(s));
    close(fd);
  } else {
    printf("FAIL\n");
  }
}

// CHECK-success: SUCCESS
// CHECK-fail: FAIL
// CHECK-test: {{^}}test{{$}}
