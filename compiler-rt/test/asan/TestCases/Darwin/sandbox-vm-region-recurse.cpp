// Check that if mach_vm_region_recurse is disallowed by sandbox, we report a message saying so.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny syscall-mig (kernel-mig-routine mach_vm_region_recurse))' %t 2>&1 | FileCheck --check-prefix=CHECK-DENY %s
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ALLOW %s
// RUN: %clangxx_asan -O3 %s -o %t
// RUN: not %run sandbox-exec -p '(version 1)(allow default)(deny syscall-mig (kernel-mig-routine mach_vm_region_recurse))' %t 2>&1 | FileCheck --check-prefix=CHECK-DENY %s
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ALLOW %s

// sandbox-exec isn't available on iOS
// UNSUPPORTED: ios

// x86_64 does not use ASAN_SHADOW_OFFSET_DYNAMIC
// UNSUPPORTED: x86_64-darwin || x86_64h-darwin

#include <stdlib.h>

int main() {
  char *x = (char *)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK-ALLOW: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK-DENY-NOT: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK-ALLOW: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-ALLOW: {{    #0 0x.* in main}}
  // CHECK-ALLOW: {{freed by thread T0 here:}}
  // CHECK-ALLOW: {{    #0 0x.* in free}}
  // CHECK-ALLOW: {{    #1 0x.* in main}}
  // CHECK-ALLOW: {{previously allocated by thread T0 here:}}
  // CHECK-ALLOW: {{    #0 0x.* in malloc}}
  // CHECK-ALLOW: {{    #1 0x.* in main}}
  // CHECK-DENY: {{.*HINT: Ensure mach_vm_region_recurse is allowed under sandbox}}
}
