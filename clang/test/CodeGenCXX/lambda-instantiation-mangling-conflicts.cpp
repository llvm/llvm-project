// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -emit-llvm -o - %t/a.cpp -verify
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -emit-llvm -o - %t/b.cpp -verify

;--- a.cpp

int normal_fn(int);
int vector_pcs_fn(int) __attribute__((aarch64_vector_pcs));

// expected-error@+3 {{definition with same mangled name '_ZZ23function_pcs_attributesvENK3$_0clIFiiEEEDaRT_' as another definition}}
__attribute__((aarch64_vector_pcs))
int function_pcs_attributes() {
  auto a = [](auto &fn) { return fn(42); };
  return a(normal_fn) + a(vector_pcs_fn);
}

;--- b.cpp

int normal_fn(int);
int streaming_fn(int) __arm_streaming;

// expected-error@+2 {{definition with same mangled name '_ZZ32function_params_normal_streamingvENK3$_0clIFiiEEEDaRT_' as another definition}}
int function_params_normal_streaming() {
  auto a = [](auto &fn) { return fn(42); };
  return a(normal_fn) + a(streaming_fn);
}
