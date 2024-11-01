// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=0 -O3 %s -o %t && \
// RUN:     not %run %t va_arg_tls >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=0 -O3 %s -o %t && \
// RUN:     not %run %t overflow >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=2 -O3 %s -o %t && \
// RUN:     not %run %t va_arg_tls >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN < %t.out

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=2 -O3 %s -o %t && \
// RUN:     not %run %t overflow >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN < %t.out

// Check that shadow and origin are passed through va_args.

// Copying origins on AArch64, LoongArch64, MIPS and PowerPC isn't supported yet.
// XFAIL: target={{(aarch64|loongarch64|mips|powerpc64).*}}

#include <stdarg.h>
#include <string.h>

__attribute__((noinline))
int sum(int n, ...) {
  va_list args;
  int i, sum = 0, arg;
  volatile int temp;
  va_start(args, n);
  for (i = 0; i < n; i++) {
    arg = va_arg(args, int);
    sum += arg;
  }
  va_end(args);
  return sum;
}

int main(int argc, char *argv[]) {
  volatile int uninit;
  volatile int a = 1, b = 2;
  if (argc == 2) {
    // Shadow/origin will be passed via va_arg_tls/va_arg_origin_tls.
    if (strcmp(argv[1], "va_arg_tls") == 0) {
      return sum(3, uninit, a, b);
    }
    // Shadow/origin of |uninit| will be passed via overflow area.
    if (strcmp(argv[1], "overflow") == 0) {
      return sum(7,
        a, a, a, a, a, a, uninit
      );
    }
  }
  return 0;
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK-ORIGIN: Uninitialized value was created by an allocation of 'uninit' in the stack frame
