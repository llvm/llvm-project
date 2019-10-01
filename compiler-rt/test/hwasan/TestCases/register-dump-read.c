// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// REQUIRES: aarch64-target-arch

// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O2 %s -o %t && not %env_hwasan_opts=fast_unwind_on_fatal=true %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -ffixed-x10 -ffixed-x20 -ffixed-x27 -O2 %s -o %t && not %env_hwasan_opts=fast_unwind_on_fatal=false %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*) malloc(10);
  asm volatile("mov x10, #0x2222\n"
               "mov x20, #0x3333\n"
               "mov x27, #0x4444\n");
  return x[16];

  // CHECK: ERROR: HWAddressSanitizer:
  // CHECK: #0 {{.*}} in main {{.*}}register-dump-read.c:[[@LINE-3]]

  // Developer note: FileCheck really doesn't like when you have a regex that
  // ends with a '}' character, e.g. the regex "[0-9]{10}" will fail, because
  // the closing '}' fails as an "unbalanced regex". We work around this by
  // encasing the trailing space after a register, or the end-of-line specifier.

  // CHECK: Registers where the failure occurred
  // CHECK-NEXT: x0{{[ ]+[0-9a-f]{16}[ ]}}x1{{[ ]+[0-9a-f]{16}[ ]}}x2{{[ ]+[0-9a-f]{16}[ ]}}x3{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x4{{[ ]+[0-9a-f]{16}[ ]}}x5{{[ ]+[0-9a-f]{16}[ ]}}x6{{[ ]+[0-9a-f]{16}[ ]}}x7{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x8{{[ ]+[0-9a-f]{16}[ ]}}x9{{[ ]+[0-9a-f]{16}[ ]}}
  // CHECK-SAME: x10 0000000000002222
  // CHECK-SAME: x11{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x12{{[ ]+[0-9a-f]{16}[ ]}}x13{{[ ]+[0-9a-f]{16}[ ]}}x14{{[ ]+[0-9a-f]{16}[ ]}}x15{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x16{{[ ]+[0-9a-f]{16}[ ]}}x17{{[ ]+[0-9a-f]{16}[ ]}}x18{{[ ]+[0-9a-f]{16}[ ]}}x19{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x20 0000000000003333
  // CHECK-SAME: x21{{[ ]+[0-9a-f]{16}[ ]}}x22{{[ ]+[0-9a-f]{16}[ ]}}x23{{[ ]+[0-9a-f]{16}$}}
  // CHECK-NEXT: x24{{[ ]+[0-9a-f]{16}[ ]}}x25{{[ ]+[0-9a-f]{16}[ ]}}x26{{[ ]+[0-9a-f]{16}[ ]}}
  // CHECK-SAME: x27 0000000000004444
  // CHECK-NEXT: x28{{[ ]+[0-9a-f]{16}[ ]}}x29{{[ ]+[0-9a-f]{16}[ ]}}x30{{[ ]+[0-9a-f]{16}$}}
}
