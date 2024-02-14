// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -S -Xclang -verify %s

// Conflicting attributes when using always_inline
__attribute__((always_inline))
void inlined_fn_streaming(void) __arm_streaming {}
// expected-error@+1 {{always_inline function 'inlined_fn_streaming' and its caller 'inlined_fn_caller' have mismatching streaming attributes}}
void inlined_fn_caller(void) __arm_streaming_compatible { inlined_fn_streaming(); }
