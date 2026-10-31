// REQUIRES: arm-registered-target
// RUN: not %clang --target=armv7r-none-eabihf -g -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: not %clang --target=armv7r-none-eabihf -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODEBUG

// DEBUG: arm-eabihf-no-fpregs.c:12:1: error: calling convention is hard-float, but floating-point registers are unavailable
// DEBUG-NEXT{LITERAL}:    12 | void hard_fn(void) {}
// DEBUG-NEXT{LITERAL}:      | ^
// NODEBUG: arm-eabihf-no-fpregs.c:12:6: error: calling convention is hard-float, but floating-point registers are unavailable
// NODEBUG-NEXT{LITERAL}:    12 | void hard_fn(void) {}
// NODEBUG-NEXT{LITERAL}:      |      ^
__attribute__((target("no-fpregs")))
void hard_fn(void) {}

// DEBUG: arm-eabihf-no-fpregs.c:23:3: error: 'soft_to_hard' calls 'hard_callee', which expects a hard-float calling convention, but floating-point registers are unavailable
// DEBUG-NEXT{LITERAL}:    23 |   hard_callee();
// DEBUG-NEXT{LITERAL}:      |   ^
// NODEBUG: arm-eabihf-no-fpregs.c:21:6: error: 'soft_to_hard' calls 'hard_callee', which expects a hard-float calling convention, but floating-point registers are unavailable
// NODEBUG-NEXT{LITERAL}:    21 | void soft_to_hard(void) {
// NODEBUG-NEXT{LITERAL}:      |      ^
__attribute__((target("no-fpregs"), pcs("aapcs")))
void soft_to_hard(void) {
  extern void hard_callee(void);
  hard_callee();
}

// DEBUG: arm-eabihf-no-fpregs.c:34:3: error: 'soft_to_indirect' makes an indirect call that expects a hard-float calling convention, but floating-point registers are unavailable
// DEBUG-NEXT{LITERAL}:    34 |   p();
// DEBUG-NEXT{LITERAL}:      |   ^
// NODEBUG: arm-eabihf-no-fpregs.c:33:6: error: 'soft_to_indirect' makes an indirect call that expects a hard-float calling convention, but floating-point registers are unavailable
// NODEBUG-NEXT{LITERAL}:    33 | void soft_to_indirect(void (*p)(void)) {
// NODEBUG-NEXT{LITERAL}:      |      ^
__attribute__((target("no-fpregs"), pcs("aapcs")))
void soft_to_indirect(void (*p)(void)) {
  p();
}
