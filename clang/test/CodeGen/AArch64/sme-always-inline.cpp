// RUN: %clang_cc1 -Waarch64-sme-attributes -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +neon -verify=expected-attr %s -DTEST_STREAMING
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +neon -verify %s -DTEST_STREAMING
// RUN: %clang_cc1 -Waarch64-sme-attributes -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +neon -verify=expected-attr %s -DTEST_COMPATIBLE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +neon -verify %s -DTEST_COMPATIBLE

#if defined(TEST_STREAMING)
#define SM_ATTR __arm_streaming
#elif defined(TEST_COMPATIBLE)
#define SM_ATTR __arm_streaming_compatible
#else
#error "Expected TEST_STREAMING or TEST_COMPATIBLE"
#endif

__attribute__((always_inline)) void incompatible_neon() {
  __attribute((vector_size(16))) char vec = { 0 };
  vec = __builtin_neon_vqaddq_v(vec, vec, 33);
}

__attribute__((always_inline)) void compatible_missing_attrs() {
    // <Empty>
}

void foo() {
    incompatible_neon();
}

// expected-note@+2 {{conflicting attribute is here}}
// expected-attr-note@+1 {{conflicting attribute is here}}
__attribute__((always_inline)) void bar() {
    incompatible_neon();
}

__attribute__((always_inline)) void baz() {
    compatible_missing_attrs();
}

void streaming_error() SM_ATTR {
    // expected-error@+3 {{always_inline function 'bar' cannot be inlined into streaming caller as it contains calls to non-streaming builtins}}
    // expected-attr-warning@+2 {{always_inline function 'bar' and its caller 'streaming_error' have mismatching streaming attributes, inlining may change runtime behaviour}}
    // expected-attr-error@+1 {{always_inline function 'bar' cannot be inlined into streaming caller as it contains calls to non-streaming builtins}}
    bar(); // -> incompatible_neon -> __builtin_neon_vqaddq_v (error)
}

void streaming_warning() SM_ATTR {
    // expected-attr-warning@+1 {{always_inline function 'baz' and its caller 'streaming_warning' have mismatching streaming attributes, inlining may change runtime behaviour}}
    baz(); // -> compatible_missing_attrs (no error)

    /// `noinline` has higher precedence than always_inline (so this is not an error)
    // expected-warning@+2 {{statement attribute 'clang::noinline' has higher precedence than function attribute 'always_inline'}}
    // expected-attr-warning@+1 {{statement attribute 'clang::noinline' has higher precedence than function attribute 'always_inline'}}
    [[clang::noinline]] bar();
}

void streaming_no_warning() SM_ATTR {
    foo(); // `foo` is not always_inline (no error/warning)
}
