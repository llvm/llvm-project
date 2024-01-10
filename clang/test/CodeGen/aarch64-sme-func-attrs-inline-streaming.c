// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -O3 -S -Xclang -verify %s

// Conflicting attributes when using always_inline
__attribute__((always_inline))
int inlined_fn_streaming(void) __arm_streaming {
    return 42;
}
// expected-error@+1 {{always_inline function 'inlined_fn_streaming' and its caller 'inlined_fn_caller' have mismatched streaming attributes}}
int inlined_fn_caller(void) { return inlined_fn_streaming(); }
__arm_locally_streaming
int inlined_fn_caller_local(void) { return inlined_fn_streaming(); }
int inlined_fn_caller_streaming(void) __arm_streaming { return inlined_fn_streaming(); }
