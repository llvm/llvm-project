// RUN: %clang_cc1 -triple thumbv7-windows -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple thumbv7-w64-windows-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple thumbv7-windows -emit-llvm -o - %s 2>&1 | FileCheck %s

// pcs("aapcs-vfp") is the default on Windows on ARM, so it is accepted
// silently. pcs("aapcs") would change the ABI and is still diagnosed.

float __attribute__((pcs("aapcs-vfp"))) with_pcs(float a, float b) { return a + b; }
float without_pcs(float a, float b) { return a + b; }

// The attributed function and a plain function have the same type.
float (*fp)(float, float) = with_pcs;

float __attribute__((pcs("aapcs"))) base_aapcs(float a, float b) { return a + b; } // expected-warning {{'pcs' calling convention is not supported for this target}}

// CHECK: define dso_local arm_aapcs_vfpcc float @with_pcs(
// CHECK: define dso_local arm_aapcs_vfpcc float @without_pcs(
