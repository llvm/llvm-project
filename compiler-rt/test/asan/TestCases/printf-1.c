// RUN: %clang_asan -O2 %s -o %t
// RUN: %env_asan_opts=check_printf=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=check_printf=0 %run %t 2>&1 | FileCheck %s
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#if defined(_WIN32)
# define snprintf _snprintf
#endif

int main() {
  volatile char c = '0';
  volatile int x = 12;
  volatile float f = 1.239;
  volatile char s[] = "34";
  // Check that printf works fine under Asan.
  printf("%c %d %.3f %s\n", c, x, f, s);
  // CHECK: 0 12 1.239 34
  // Check that snprintf works fine under Asan.
  char buf[4];
  snprintf(buf, 1000, "qwe");
  printf("%s\n", buf);
  // CHECK: qwe
  return 0;
}
