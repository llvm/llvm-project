// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -S -Xclang -verify %s

// Conflicting attributes when using always_inline
#define __ai __attribute__((always_inline))
__ai void inlined_fn_streaming_compatible(void) __arm_streaming_compatible {}
__ai void inlined_fn(void) {}
void inlined_fn_caller(void) { inlined_fn_streaming_compatible(); }
__arm_locally_streaming
void inlined_fn_caller_local(void) { inlined_fn_streaming_compatible(); }
void inlined_fn_caller_streaming(void) __arm_streaming { inlined_fn_streaming_compatible(); }
// expected-error@+1 {{always_inline function 'inlined_fn' and its caller 'inlined_fn_caller_compatible' have mismatching streaming attributes}}
void inlined_fn_caller_compatible(void) __arm_streaming_compatible { inlined_fn(); }
