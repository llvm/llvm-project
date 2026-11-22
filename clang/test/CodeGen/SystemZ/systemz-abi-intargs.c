// RUN: %clang_cc1 -triple s390x-linux-gnu -S -O3 %s -o - \
// RUN:    -mllvm -debug-only=systemz-lower 2>&1 | FileCheck %s
//
// REQUIRES: systemz-registered-target, asserts
//
// Check that clang verifies the extensions of narrow integer arguments by default.

int __attribute__ ((noinline)) foo(int Arg, unsigned Arg2) {
  return Arg + Arg2;
}

int fun(short *Arg, unsigned char *Arg2) {
  return foo(*Arg, *Arg2);
}

// CHECK: Return argument verified as ABI compliant        : noundef signext i32 @foo(i32 signext, i32 zeroext)
// CHECK: Outgoing call arguments verified as ABI compliant: noundef signext i32 @foo(i32 signext, i32 zeroext)

// CHECK-LABEL: foo:
// CHECK:       ar   %r3, %r2
// CHECK-NEXT:  lgfr %r2, %r3
// CHECK-NEXT:  br   %r14
//
// CHECK-LABEL: fun:
// CHECK:       lgh  %r2, 0(%r2)
// CHECK-NEXT:  llgc %r3, 0(%r3)
// CHECK-NEXT:  jg   foo@PLT

