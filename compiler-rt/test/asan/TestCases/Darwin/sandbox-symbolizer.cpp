// In a non-forking sandbox, we can't spawn an external symbolizer, but dladdr()
// should still work and provide function names. No line numbers though.
// Second, `atos` symbolizer can't inspect a process that has an inaccessible
// task port, in which case we should again fallback to dladdr gracefully.

// RUN: %clangxx_asan -O0 %s -o %t 
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny process-fork)' %t 2>&1 | FileCheck %s
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny process-exec (literal "/usr/bin/atos"))' %t 2>&1 | FileCheck %s
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny mach-priv-task-port)' %t 2>&1 | FileCheck %s
// RUN: env ASAN_SYMBOLIZER_PATH="" not %run sandbox-exec -p '(version 1)(allow default)(deny mach-priv-task-port)' %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t 
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny process-fork)' %t 2>&1 | FileCheck %s
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny process-exec (literal "/usr/bin/atos"))' %t 2>&1 | FileCheck %s
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny mach-priv-task-port)' %t 2>&1 | FileCheck %s
// RUN: env ASAN_SYMBOLIZER_PATH="" not %run sandbox-exec -p '(version 1)(allow default)(deny mach-priv-task-port)' %t 2>&1 | FileCheck %s

// sandbox-exec isn't available on iOS
// UNSUPPORTED: ios

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main}}
  // CHECK: {{freed by thread T0 here:}}
  // CHECK: {{    #0 0x.* in free}}
  // CHECK: {{    #1 0x.* in main}}
  // CHECK: {{previously allocated by thread T0 here:}}
  // CHECK: {{    #0 0x.* in malloc}}
  // CHECK: {{    #1 0x.* in main}}
}
