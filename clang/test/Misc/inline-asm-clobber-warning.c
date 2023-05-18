/// This test checks that the warning includes the location in the C source
/// file that contains the inline asm. Instead of saying <inline asm> for both.
/// Although this warning is emitted in llvm it cannot be tested from IR as
/// it does not have that location information at that stage.

// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -c %s -o /dev/null \
// RUN:   2>&1 | FileCheck %s

// REQUIRES: arm-registered-target

void bar(void) {
  __asm__ __volatile__("nop"
                       :
                       :
                       : "sp");
}

// CHECK:      inline-asm-clobber-warning.c:12:24: warning: inline asm clobber list contains reserved registers: SP [-Winline-asm]
// CHECK-NEXT:     __asm__ __volatile__("nop"
// CHECK-NEXT:                          ^
// CHECK-NEXT: inline-asm-clobber-warning.c:12:24: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
