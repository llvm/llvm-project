// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +tme -verify %s
void t_cancel_const(unsigned short u) {
  __builtin_arm_tcancel(u); // expected-error{{argument to '__builtin_arm_tcancel' must be a constant integer}}
}

// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +tme -verify %s
void t_cancel_range(void) {
  __builtin_arm_tcancel(0x12345u); // expected-error{{argument value 74565 is outside the valid range [0, 65535]}}
}
