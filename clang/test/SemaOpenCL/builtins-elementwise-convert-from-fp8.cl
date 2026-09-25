// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL1.2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -cl-ext=-cl_khr_fp16 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef unsigned char uchar3 __attribute__((ext_vector_type(3)));
typedef _Float16 float16_3 __attribute__((ext_vector_type(3)));
typedef __bf16 bfloat16_3 __attribute__((ext_vector_type(3)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef half half3 __attribute__((ext_vector_type(3)));

// The f16 suffix always produces _Float16, including when cl_khr_fp16 is
// disabled. OpenCL half is a distinct type.
void results(unsigned char bits, uchar3 packed) {
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m2_f16(bits)),
                     _Float16), "");
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits)),
                     __bf16), "");
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits)),
                     float), "");
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m2_f16(packed)),
                     float16_3), "");
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e4m3fn_bf16(packed)),
                     bfloat16_3), "");
  _Static_assert(__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m3fnu_f32(packed)),
                     float3), "");
  _Static_assert(!__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m2_f16(bits)),
                     half), "");
  _Static_assert(!__builtin_types_compatible_p(
                     __typeof__(__builtin_elementwise_convert_from_f8e5m2_f16(packed)),
                     half3), "");

  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits);
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(packed);
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(packed);
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(packed);
}
