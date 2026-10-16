// RUN: %clang_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=check_printf=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: freebsd || netbsd

#include <unistd.h>

int main() {
  const char fmt[2] = "%s%d\n";
  setproctitle("%s", "setproctitle");
  setproctitle("%c%c%c", 'c', "a", 'e');
  setproctitle(fmt, "abcdef", -5);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow
// CHECK-NEXT: READ of size {{[0-9]+}} at {{.*}}
// CHECK: #0 {{.*}} printf_common
// CHECK: #1 {{.*}} setproctitle
// CHECK: #2 {{.*}} main
