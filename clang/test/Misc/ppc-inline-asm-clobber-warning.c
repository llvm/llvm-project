/// This test checks that the warning includes the location in the C source
/// file that contains the inline asm. Although this warning is emitted in llvm
/// it cannot be tested from IR as it does not have that location information at
/// that stage.

// REQUIRES: powerpc-registered-target

// RUN: %clang --target=powerpc-unknown-unknown -mcpu=pwr7 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64-unknown-unknown -mcpu=pwr7 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s

void test_r1_clobber() {
  __asm__("nop":::"r1");
}

// CHECK:      ppc-inline-asm-clobber-warning.c:14:11: warning: inline asm clobber list contains reserved registers: R1 [-Winline-asm]
// CHECK-NEXT:   __asm__("nop":::"r1");
// CHECK-NEXT:           ^
// CHECK-NEXT: ppc-inline-asm-clobber-warning.c:14:11: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

void test_1_clobber() {
  __asm__("nop":::"1");
}

// CHECK:      ppc-inline-asm-clobber-warning.c:23:11: warning: inline asm clobber list contains reserved registers: R1 [-Winline-asm]
// CHECK-NEXT:   __asm__("nop":::"1");
// CHECK-NEXT:           ^
// CHECK-NEXT: ppc-inline-asm-clobber-warning.c:23:11: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

void test_sp_clobber() {
  __asm__("nop":::"sp");
}

// CHECK:      ppc-inline-asm-clobber-warning.c:32:11: warning: inline asm clobber list contains reserved registers: R1 [-Winline-asm]
// CHECK-NEXT:   __asm__("nop":::"sp");
// CHECK-NEXT:           ^
// CHECK-NEXT: ppc-inline-asm-clobber-warning.c:32:11: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
