// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -O3 -S -Xclang -verify %s

// Conflicting attributes when using always_inline
__attribute__((always_inline)) __arm_locally_streaming
int inlined_fn_local(void) {
    return 42;
}
// expected-error@+1 {{always_inline function 'inlined_fn_local' and its caller 'inlined_fn_caller' have mismatched streaming attributes}}
int inlined_fn_caller(void) { return inlined_fn_local(); }
__arm_locally_streaming
int inlined_fn_caller_local(void) { return inlined_fn_local(); }
int inlined_fn_caller_streaming(void) __arm_streaming { return inlined_fn_local(); }
