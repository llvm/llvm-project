// The same boundary crossings as arm64ec-fp80-thunks.ll, with `long double` in
// its 64-bit form.  It is then an ordinary `double`, which the x64 side passes
// in an XMM register and the ARM64EC side in the very same physical register,
// so the thunks carry the value through untouched and none of the x87 helpers
// takes part.
//
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-64 -O2 -S -o - %s | FileCheck %s

static long double ec_helper(long double x) { return x + 1.0L; }

long double ec_identity(long double x) { return ec_helper(x); }

extern long double x64_callee(void);
long double ec_forward(void) { return x64_callee(); }

extern void x64_take(long double);
void ec_hand_off(long double x) { x64_take(x); }

// An ARM64EC call runs on hardware double precision, so the helper folds into
// a plain addition on `d` registers; no 80-bit arithmetic is involved.
// CHECK-LABEL: "#ec_identity":
// CHECK:       fmov    d1, #1.00000000
// CHECK-NEXT:  fadd    d0, d0, d1
// CHECK-NEXT:  ret

// The entry thunk receives the value in xmm0, which is the register the
// ARM64EC side expects it in, so it is neither loaded nor stored.
// CHECK-LABEL: $ientry_thunk$cdecl$d$d:
// CHECK-NOT:   ldr
// CHECK:       blr     x9

// The exit thunk returns the value the same way, again moving nothing.
// CHECK-LABEL: $iexit_thunk$cdecl$d$v:
// CHECK-NOT:   str
// CHECK:       blr     x16
