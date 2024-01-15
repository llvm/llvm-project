// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -O3 -S -Xclang -verify %s

// Conflicting attributes when using always_inline
__attribute__((always_inline))
int inlined_fn_streaming_compatible(void) __arm_streaming_compatible {
    return 42;
}
__attribute__((always_inline))
int inlined_fn(void) {
    return 42;
}
int inlined_fn_caller(void) { return inlined_fn_streaming_compatible(); }
__arm_locally_streaming
int inlined_fn_caller_local(void) { return inlined_fn_streaming_compatible(); }
int inlined_fn_caller_streaming(void) __arm_streaming { return inlined_fn_streaming_compatible(); }
// expected-error@+1 {{always_inline function 'inlined_fn' and its caller 'inlined_fn_caller_compatible' have mismatched streaming attributes}}
int inlined_fn_caller_compatible(void) __arm_streaming_compatible { return inlined_fn(); }
