// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -S -Xclang -verify %s

// Conflicting attributes when using always_inline
__attribute__((always_inline)) __arm_locally_streaming
void inlined_fn_local(void) {}
__arm_locally_streaming
void inlined_fn_caller_local(void) { inlined_fn_local(); }
void inlined_fn_caller_streaming(void) __arm_streaming { inlined_fn_local(); }
// expected-error@+1 {{always_inline function 'inlined_fn_local' and its caller 'inlined_fn_caller' have mismatching streaming attributes}}
void inlined_fn_caller(void) { inlined_fn_local(); }
