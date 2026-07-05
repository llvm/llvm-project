// UNSUPPORTED: ios
// We can reduce the scope of this test to check that we set the crash reporter
// buffers correctly instead of reading from the crashlog.
// For now, disable this test.
// REQUIRES: rdar_74544282
// REQUIRES: expensive
// Check that ASan reports on OS X actually crash the process (abort_on_error=1)
// and that crash is logged via the crash reporter with ASan logs in the
// Application Specific Information section of the log.

// RUN: %clangxx_asan %s -o %t

// crash hard so the crashlog is created.
// RUN: %env_asan_opts=abort_on_error=1 not --crash %run %t > %t.process_output.txt 2>&1
// RUN: %print_crashreport_for_pid --binary-filename=%basename_t.tmp \
// RUN: --pid=$(%get_pid_from_output --infile=%t.process_output.txt) \
// RUN: | FileCheck %s --check-prefixes CHECK-CRASHLOG

#include <stdlib.h>
int main() {
  char *x = (char *)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // needs to crash hard so the crashlog exists...
  // CHECK-CRASHLOG: {{.*Application Specific Information:}}
  // CHECK-CRASHLOG-NEXT: {{=====}}
  // CHECK-CRASHLOG-NEXT: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK-CRASHLOG: {{abort()}}
}
