// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -emit-llvm -o - %s -verify -DTEST1
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -emit-llvm -o - %s -verify -DTEST2

int normal_fn(int);
int streaming_fn(int) __arm_streaming;
int vector_pcs_fn(int) __attribute__((aarch64_vector_pcs));

#ifdef TEST1

// expected-error@+4 {{definition with same mangled name '_ZZ23function_pcs_attributesvENK3$_0clIFiiEEEDaRT_' as another definition}}
// expected-note@+3 {{previous definition is here}}
__attribute__((aarch64_vector_pcs))
int function_pcs_attributes() {
  auto a = [](auto &fn) { return fn(42); };
  return a(normal_fn) + a(vector_pcs_fn);
}

#endif

#ifdef TEST2

// expected-error@+3 {{definition with same mangled name '_ZZ32function_params_normal_streamingvENK3$_0clIFiiEEEDaRT_' as another definition}}
// expected-note@+2 {{previous definition is here}}
int function_params_normal_streaming() {
  auto a = [](auto &fn) { return fn(42); };
  return a(normal_fn) + a(streaming_fn);
}

#endif
