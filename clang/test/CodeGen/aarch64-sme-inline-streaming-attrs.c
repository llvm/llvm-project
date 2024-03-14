// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -verify -DTEST_NONE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -verify -DTEST_COMPATIBLE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -verify -DTEST_STREAMING %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -verify -DTEST_LOCALLY %s

#define __ai __attribute__((always_inline))
__ai void inlined_fn(void) {}
__ai void inlined_fn_streaming_compatible(void) __arm_streaming_compatible {}
__ai void inlined_fn_streaming(void) __arm_streaming {}
__ai __arm_locally_streaming void inlined_fn_local(void) {}

#ifdef TEST_NONE
void caller(void) {
    inlined_fn();
    inlined_fn_streaming_compatible();
    inlined_fn_streaming(); // expected-error {{always_inline function 'inlined_fn_streaming' and its caller 'caller' have mismatching streaming attributes}}
    inlined_fn_local(); // expected-error {{always_inline function 'inlined_fn_local' and its caller 'caller' have mismatching streaming attributes}}
}
#endif

#ifdef TEST_COMPATIBLE
void caller_compatible(void) __arm_streaming_compatible {
    inlined_fn(); // expected-error {{always_inline function 'inlined_fn' and its caller 'caller_compatible' have mismatching streaming attributes}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming(); // expected-error {{always_inline function 'inlined_fn_streaming' and its caller 'caller_compatible' have mismatching streaming attributes}}
    inlined_fn_local(); // expected-error {{always_inline function 'inlined_fn_local' and its caller 'caller_compatible' have mismatching streaming attributes}}
}
#endif

#ifdef TEST_STREAMING
void caller_streaming(void) __arm_streaming {
    inlined_fn(); // expected-error {{always_inline function 'inlined_fn' and its caller 'caller_streaming' have mismatching streaming attributes}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming();
    inlined_fn_local();
}
#endif

#ifdef TEST_LOCALLY
__arm_locally_streaming
void caller_local(void) {
    inlined_fn(); // expected-error {{always_inline function 'inlined_fn' and its caller 'caller_local' have mismatching streaming attributes}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming();
    inlined_fn_local();
}
#endif
