// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -Waarch64-sme-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -Waarch64-sme-attributes -fsyntax-only -verify=expected-cpp -x c++ %s

__attribute__((target_clones("sve", "simd"))) void ok_arm_streaming(void) __arm_streaming {}
__arm_locally_streaming __attribute__((target_version("sme2"))) void ok_arm_streaming(void) __arm_streaming {}
__attribute__((target_version("default"))) void ok_arm_streaming(void) __arm_streaming {}

__attribute__((target_clones("sve", "simd"))) void ok_arm_streaming_compatible(void) __arm_streaming_compatible {}
__arm_locally_streaming __attribute__((target_version("sme2"))) void ok_arm_streaming_compatible(void) __arm_streaming_compatible {}
__attribute__((target_version("default"))) void ok_arm_streaming_compatible(void) __arm_streaming_compatible {}

__arm_locally_streaming __attribute__((target_clones("sve", "simd"))) void ok_no_streaming(void) {}
__attribute__((target_version("sme2"))) void ok_no_streaming(void) {}
__attribute__((target_version("default"))) void ok_no_streaming(void) {}

__attribute__((target_clones("sve", "simd"))) void bad_mixed_streaming(void) {}
// expected-cpp-error@+2 {{multiversioned function declaration has a different calling convention}}
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
__attribute__((target_version("sme2"))) void bad_mixed_streaming(void) __arm_streaming {}
// expected-cpp-error@+2 {{multiversioned function declaration has a different calling convention}}
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
__attribute__((target_version("default"))) void bad_mixed_streaming(void) __arm_streaming_compatible {}
// expected-cpp-error@+2 {{multiversioned function declaration has a different calling convention}}
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
__arm_locally_streaming __attribute__((target_version("dotprod"))) void bad_mixed_streaming(void) __arm_streaming {}

void n_caller(void) {
  ok_arm_streaming();
  ok_arm_streaming_compatible();
  ok_no_streaming();
  bad_mixed_streaming();
}

void s_caller(void) __arm_streaming {
  ok_arm_streaming();
  ok_arm_streaming_compatible();
  ok_no_streaming();
  bad_mixed_streaming();
}

void sc_caller(void) __arm_streaming_compatible {
  ok_arm_streaming();
  ok_arm_streaming_compatible();
  ok_no_streaming();
  bad_mixed_streaming();
}
