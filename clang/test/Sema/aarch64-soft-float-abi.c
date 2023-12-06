// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +fp-armv8 -fsyntax-only -target-abi aapcs      -verify=fp-hard %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp-armv8 -fsyntax-only -target-abi aapcs-soft -verify=nofp-soft %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp-armv8 -fsyntax-only -target-abi aapcs      -verify=nofp-hard %s
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

float test_float_ret(void) { return 3.141f; } // #1
// nofp-hard-error@#1 {{'test_float_ret' requires 'float' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#1 {{'test_float_ret' defined here}}

struct HFA test_hfa_ret(void) { return (struct HFA){}; } // #2
// nofp-hard-error@#2 {{'test_hfa_ret' requires 'struct HFA' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#2 {{'test_hfa_ret' defined here}}

struct non_HFA test_non_hfa_ret(void) { return (struct non_HFA){}; } // #3
// nofp-hard-error@#3 {{'test_non_hfa_ret' requires 'struct non_HFA' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#3 {{'test_non_hfa_ret' defined here}}

void test_float_arg(float a) {} // #4
// nofp-hard-error@#4 {{'test_float_arg' requires 'float' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#4 {{'test_float_arg' defined here}}

void test_hfa_arg(struct HFA a) {} // #5
// nofp-hard-error@#5 {{'test_hfa_arg' requires 'struct HFA' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#5 {{'test_hfa_arg' defined here}}

void test_non_hfa_arg(struct non_HFA a) {} // #6
// nofp-hard-error@#6 {{'test_non_hfa_arg' requires 'struct non_HFA' type support, but ABI 'aapcs' does not support it}}
// nofp-hard-note@#6 {{'test_non_hfa_arg' defined here}}

int test_expr_float(int a) { return a + 1.0f; } // #7
// nofp-hard-error@#7 {{expression requires 'float' type support, but ABI 'aapcs' does not support it}}

int test_expr_double(int a) { return a + 1.0; } // #8
// nofp-hard-error@#8 {{expression requires 'double' type support, but ABI 'aapcs' does not support it}}
