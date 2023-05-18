// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +fp-armv8 -S -o /dev/null -target-abi aapcs      -verify=fp-hard %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp-armv8 -S -o /dev/null -target-abi aapcs-soft -verify=nofp-soft %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp-armv8 -S -o /dev/null -target-abi aapcs      -verify=nofp-hard %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp-armv8 -S -o /dev/null -target-abi aapcs -O1  -verify=nofp-hard,nofp-hard-opt -emit-llvm %s
// No run line needed for soft-float ABI with an FPU because that is rejected by the driver

// With the hard-float ABI and a target with an FPU, FP arguments are passed in
// FP registers, no diagnostics needed.
// fp-hard-no-diagnostics

// With the soft-float ABI, FP arguments are passed in integer registers, no
// diagnostics needed.
// nofp-soft-no-diagnostics

// With the hard-float ABI but no FPU, FP arguments cannot be passed in an
// ABI-compatible way, so we report errors for these cases:

struct HFA {
  float x, y;
};

struct non_HFA {
  float x;
  int y;
};

// Floating-point arguments are returns are rejected
void test_fp16_arg(__fp16 a) {}
// nofp-hard-error@-1 {{'a' requires '__fp16' type support, but ABI 'aapcs' does not support it}}
__fp16 test_fp16_ret(void) { return 3.141; }
// nofp-hard-error@-1 {{'test_fp16_ret' requires '__fp16' type support, but ABI 'aapcs' does not support it}}
void test_float_arg(float a) {}
// nofp-hard-error@-1 {{'a' requires 'float' type support, but ABI 'aapcs' does not support it}}
float test_float_ret(void) { return 3.141f; }
// nofp-hard-error@-1 {{'test_float_ret' requires 'float' type support, but ABI 'aapcs' does not support it}}
void test_double_arg(double a) {}
// nofp-hard-error@-1 {{'a' requires 'double' type support, but ABI 'aapcs' does not support it}}
double test_double_ret(void) { return 3.141; }
// nofp-hard-error@-1 {{'test_double_ret' requires 'double' type support, but ABI 'aapcs' does not support it}}
void test_long_double_arg(long double a) {}
// nofp-hard-error@-1 {{'a' requires 'long double' type support, but ABI 'aapcs' does not support it}}
long double test_long_double_ret(void) { return 3.141L; }
// nofp-hard-error@-1 {{'test_long_double_ret' requires 'long double' type support, but ABI 'aapcs' does not support it}}

// HFAs would be passed in floating-point registers, so are rejected.
void test_hfa_arg(struct HFA a) {}
// nofp-hard-error@-1 {{'a' requires 'struct HFA' type support, but ABI 'aapcs' does not support it}}
struct HFA test_hfa_ret(void) { return (struct HFA){}; }
// nofp-hard-error@-1 {{'test_hfa_ret' requires 'struct HFA' type support, but ABI 'aapcs' does not support it}}

// Note: vector types cannot be created at all for targets without an FPU, so
// it is not possible to create a function which passes/returns them when using
// either the default or soft-float ABI. This is tested elsewhere.

// This struct contains a floating-point type, but is not an HFA, so can be
// passed/returned without affecting the ABI.
struct non_HFA test_non_hfa_ret(void) { return (struct non_HFA){}; }
void test_non_hfa_arg(struct non_HFA a) {}

// This inline function does not get code-generated because there is no use of
// it in this file, so we we don't emit an error for it, matching GCC's
// behaviour.
inline void test_float_arg_inline(float a) {}

// This inline function is used, so we emit the error if we generate code for
// it. The code isn't generated at -O0, so no error is emitted there.
inline void test_float_arg_inline_used(float a) {}
// nofp-hard-opt-error@-1 {{'a' requires 'float' type support, but ABI 'aapcs' does not support it}}
void use_inline() { test_float_arg_inline_used(1.0f); }

// The always_inline attribute causes an inline function to always be
// code-genned, even at -O0, so we always emit the error.
__attribute((always_inline))
inline void test_float_arg_always_inline_used(float a) {}
// nofp-hard-error@-1 {{'a' requires 'float' type support, but ABI 'aapcs' does not support it}}
void use_always_inline() { test_float_arg_always_inline_used(1.0f); }

// Floating-point expressions, global variables and local variables do not
// affect the ABI, so are allowed. GCC does reject some uses of floating point
// types like this, but it does so after optimisation, which we can't
// accurately match in clang.
int test_expr_float(int a) { return a + 1.0f; }
int test_expr_double(int a) { return a + 1.0; }

float global_float = 2.0f * 3.5f;
float global_double = 2.0 * 3.5;

int test_var_float(int a) {
  float f = a;
  f *= 6.0;
  return (int)f;
}
int test_var_double(int a) {
  double d = a;
  d *= 6.0;
  return (int)d;
}
