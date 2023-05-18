// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +v8a -fsyntax-only -verify -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +d128 -fsyntax-only -verify=d128 -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct
// system-register intriniscs.

// All the calls below are valid if you have -target-feature +d128
// d128-no-diagnostics

#include <arm_acle.h>

void anytarget(void) {
  unsigned x = __arm_rsr("1:2:3:4:5");
  __arm_wsr("1:2:3:4:5", x);
  unsigned long y = __arm_rsr64("1:2:3:4:5");
  __arm_wsr64("1:2:3:4:5", y);
  void *p = __arm_rsrp("1:2:3:4:5");
  __arm_wsrp("1:2:3:4:5", p);
}

__attribute__((target("d128")))
void d128target(void) {
  __uint128_t x = __arm_rsr128("1:2:3:4:5");
  __arm_wsr128("1:2:3:4:5", x);
}

void notd128target(void) {
  __uint128_t x = __arm_rsr128("1:2:3:4:5"); // expected-error {{needs target feature d128}}
  __arm_wsr128("1:2:3:4:5", x); // expected-error {{needs target feature d128}}
}
