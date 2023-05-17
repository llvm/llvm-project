// RUN: %clang_asan -O2 %s -o %t
// RUN: %env_asan_opts=check_printf=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ON %s
// RUN: %env_asan_opts=check_printf=0 %run %t 2>&1 | FileCheck --check-prefix=CHECK-OFF %s
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ON %s

// FIXME: printf is not intercepted on Windows yet.
// XFAIL: target={{.*windows-(msvc.*|gnu)}}

// New Bionic rejects %n
// https://android.googlesource.com/platform/bionic/+/41398d03b7e8e0dfb951660ae713e682e9fc0336
// UNSUPPORTED: android

#include <stdio.h>
int main() {
#ifdef _MSC_VER
  // FIXME: The test raises a dialog even though it's XFAILd.
  return 42;
#endif
  volatile char c = '0';
  volatile int x = 12;
  volatile float f = 1.239;
  volatile char s[] = "34";
  volatile int n[1];
  printf("%c %d %.3f %s%n\n", c, x, f, s, &n[1]);
  return 0;
  // Check that %n is sanitized.
  // CHECK-ON: stack-buffer-overflow
  // CHECK-ON-NOT: 0 12 1.239 34
  // CHECK-OFF: 0 12 1.239 34
}
