// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature -sse2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature +sse2 -fsyntax-only -verify=supported %s
// supported-no-diagnostics

// Recognition of the spelling does not imply support for the destination type.
#if !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_bf16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f32) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_f16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_bf16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_f32) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_f16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_bf16) || \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_f32)
#error missing encoded floating-point conversion builtin
#endif

typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));

void scalar(unsigned char bits) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits); // expected-error {{_Float16 is not supported on this target}}
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits); // expected-error {{__bf16 is not supported on this target}}
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

void vector(uchar4 bits) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits); // expected-error {{_Float16 is not supported on this target}}
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits); // expected-error {{__bf16 is not supported on this target}}
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

void unevaluated(unsigned char bits, uchar4 packed) {
  (void)sizeof(__builtin_elementwise_convert_from_f8e5m2_f16(bits)); // expected-error {{_Float16 is not supported on this target}}
  (void)sizeof(__builtin_elementwise_convert_from_f8e4m3fn_bf16(packed)); // expected-error {{__bf16 is not supported on this target}}
}
