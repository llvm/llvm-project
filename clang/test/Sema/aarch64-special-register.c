// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify -triple aarch64 %s

void string_literal(unsigned v) {
  __builtin_arm_wsr(0, v); // expected-error {{expression is not a string literal}}
}

void wsr_1(unsigned v) {
  __builtin_arm_wsr("sysreg", v);
}

void wsrp_1(void *v) {
  __builtin_arm_wsrp("sysreg", v);
}

void wsr64_1(unsigned long v) {
  __builtin_arm_wsr64("sysreg", v);
}

void wsr128_1(__uint128_t v) {
  __builtin_arm_wsr128("sysreg", v);
}

unsigned rsr_1(void) {
  return __builtin_arm_rsr("sysreg");
}

void *rsrp_1(void) {
  return __builtin_arm_rsrp("sysreg");
}

unsigned long rsr64_1(void) {
  return __builtin_arm_rsr64("sysreg");
}

__uint128_t rsr128_1(void) {
  return __builtin_arm_rsr128("sysreg");
}

void wsr_2(unsigned v) {
  __builtin_arm_wsr("0:1:2:3:4", v);
}

void wsrp_2(void *v) {
  __builtin_arm_wsrp("0:1:2:3:4", v);
}

void wsr64_2(unsigned long v) {
  __builtin_arm_wsr64("0:1:2:3:4", v);
}

unsigned rsr_2(void) {
  return __builtin_arm_rsr("0:1:15:15:4");
}

void *rsrp_2(void) {
  return __builtin_arm_rsrp("0:1:2:3:4");
}

unsigned long rsr64_2(void) {
  return __builtin_arm_rsr64("0:1:15:15:4");
}

__uint128_t rsr128_2(void) {
  return __builtin_arm_rsr128("0:1:15:15:4");
}

void wsr_3(unsigned v) {
  __builtin_arm_wsr("0:1:2", v); //expected-error {{invalid special register for builtin}}
}

void wsrp_3(void *v) {
  __builtin_arm_wsrp("0:1:2", v); //expected-error {{invalid special register for builtin}}
}

void wsr64_3(unsigned long v) {
  __builtin_arm_wsr64("0:1:2", v); //expected-error {{invalid special register for builtin}}
}

void wsr128_3(__uint128_t v) {
  __builtin_arm_wsr128("0:1:2", v); //expected-error {{invalid special register for builtin}}
}

unsigned rsr_3(void) {
  return __builtin_arm_rsr("0:1:2"); //expected-error {{invalid special register for builtin}}
}

unsigned rsr_4(void) {
  return __builtin_arm_rsr("0:1:2:3:8"); //expected-error {{invalid special register for builtin}}
}

unsigned rsr_5(void) {
  return __builtin_arm_rsr("0:8:1:2:3"); //expected-error {{invalid special register for builtin}}
}

unsigned rsr_6(void) {
  return __builtin_arm_rsr("0:1:16:16:2"); //expected-error {{invalid special register for builtin}}
}

void *rsrp_3(void) {
  return __builtin_arm_rsrp("0:1:2"); //expected-error {{invalid special register for builtin}}
}

unsigned long rsr64_3(void) {
  return __builtin_arm_rsr64("0:1:2"); //expected-error {{invalid special register for builtin}}
}

unsigned long rsr64_4(void) {
  return __builtin_arm_rsr64("0:1:2:3:8"); //expected-error {{invalid special register for builtin}}
}

unsigned long rsr64_5(void) {
  return __builtin_arm_rsr64("0:8:2:3:4"); //expected-error {{invalid special register for builtin}}
}

unsigned long rsr64_6(void) {
  return __builtin_arm_rsr64("0:1:16:16:2"); //expected-error {{invalid special register for builtin}}
}

__uint128_t rsr128_3(void) {
  return __builtin_arm_rsr128("0:1:2"); //expected-error {{invalid special register for builtin}}
}

__uint128_t rsr128_4(void) {
  return __builtin_arm_rsr128("0:1:2:3:8"); //expected-error {{invalid special register for builtin}}
}

__uint128_t rsr128_5(void) {
  return __builtin_arm_rsr128("0:8:2:3:4"); //expected-error {{invalid special register for builtin}}
}

__uint128_t rsr128_6(void) {
  return __builtin_arm_rsr128("0:1:16:16:2"); //expected-error {{invalid special register for builtin}}
}

void wsr_4(void) {
  __builtin_arm_wsr("spsel", 15);
  __builtin_arm_wsr("daifclr", 15);
  __builtin_arm_wsr("daifset", 15);
  __builtin_arm_wsr("pan", 15);
  __builtin_arm_wsr("uao", 15);
  __builtin_arm_wsr("dit", 15);
  __builtin_arm_wsr("ssbs", 15);
  __builtin_arm_wsr("tco", 15);

  __builtin_arm_wsr("allint", 1);
  __builtin_arm_wsr("pm", 1);
}

void wsr64_4(void) {
  __builtin_arm_wsr("spsel", 15);
  __builtin_arm_wsr("daifclr", 15);
  __builtin_arm_wsr("daifset", 15);
  __builtin_arm_wsr("pan", 15);
  __builtin_arm_wsr("uao", 15);
  __builtin_arm_wsr("dit", 15);
  __builtin_arm_wsr("ssbs", 15);
  __builtin_arm_wsr("tco", 15);

  __builtin_arm_wsr("allint", 1);
  __builtin_arm_wsr("pm", 1);
}

void wsr_5(unsigned v) {
  __builtin_arm_wsr("spsel", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("daifclr", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("daifset", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("pan", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("uao", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("dit", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("ssbs", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("tco", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("allint", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr("pm", v); // expected-error {{must be a constant integer}}
}

void wsr64_5(unsigned long v) {
  __builtin_arm_wsr64("spsel", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("daifclr", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("daifset", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("pan", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("uao", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("dit", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("ssbs", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("tco", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("allint", v); // expected-error {{must be a constant integer}}
  __builtin_arm_wsr64("pm", v); // expected-error {{must be a constant integer}}
}

void wsr_6(void) {
  __builtin_arm_wsr("spsel", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("daifclr", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("daifset", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("pan", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("uao", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("dit", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("ssbs", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("tco", 16); // expected-error {{outside the valid range}}

  __builtin_arm_wsr("allint", 2); // expected-error {{outside the valid range}}
  __builtin_arm_wsr("pm", 2); // expected-error {{outside the valid range}}
}

void wsr64_6(void) {
  __builtin_arm_wsr64("spsel", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("daifclr", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("daifset", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("pan", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("uao", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("dit", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("ssbs", 16); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("tco", 16); // expected-error {{outside the valid range}}

  __builtin_arm_wsr64("allint", 2); // expected-error {{outside the valid range}}
  __builtin_arm_wsr64("pm", 2); // expected-error {{outside the valid range}}
}
