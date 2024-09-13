// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -emit-llvm -o - %s -verify -DTEST1
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -emit-llvm -o - %s -verify -DTEST2

int normal_fn(int);
int streaming_fn(int) __arm_streaming;
int streaming_compatible_fn(int) __arm_streaming_compatible;

#ifdef TEST1

// expected-error@+3 {{definition with same mangled name '_ZZ32function_params_normal_streamingvENK3$_0clIFiiEEEDaRT_' as another definition}}
// expected-note@+2 {{previous definition is here}}
int function_params_normal_streaming() {
  auto a = [](auto &fn) { return fn(42); };
  return a(normal_fn) + a(streaming_fn);
}

#endif

#ifdef TEST2

// expected-error@+3 {{definition with same mangled name '_ZZ36function_params_streaming_compatiblevENK3$_0clIFiiEEEDaRT_' as another definition}}
// expected-note@+2 {{previous definition is here}}
int function_params_streaming_compatible() {
  auto a = [](auto &fn) { return fn(42); };
  return a(streaming_fn) + a(streaming_compatible_fn);
}

#endif
