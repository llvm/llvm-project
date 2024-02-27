// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct SME intrinsics.

#include <arm_sme.h>

__attribute__((target("sme")))
void test_sme(svbool_t pg, void *ptr) __arm_streaming __arm_inout("za") {
  svld1_hor_za8(0, 0, pg, ptr);
}

__attribute__((target("arch=armv8-a+sme")))
void test_arch_sme(svbool_t pg, void *ptr) __arm_streaming __arm_inout("za") {
  svld1_hor_vnum_za32(0, 0, pg, ptr, 0);
}

__attribute__((target("+sme")))
void test_plus_sme(svbool_t pg, void *ptr) __arm_streaming __arm_inout("za") {
  svst1_ver_za16(0, 0, pg, ptr);
}

__attribute__((target("+sme")))
void undefined(svbool_t pg, void *ptr) __arm_inout("za") {
  svst1_ver_vnum_za64(0, 0, pg, ptr, 0); // expected-warning {{builtin call has undefined behaviour when called from a non-streaming function}}
}
