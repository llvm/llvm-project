// RUN: %clang_cc1 -triple spir-unknown-unknown -x cl \
// RUN:   -finclude-default-header -fsyntax-only -verify %s
// expected-no-diagnostics

#if !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f32)
#error "missing elementwise arbitrary FP conversion builtin"
#endif

#if __has_builtin(__builtin_elementwise_convert_from_f6e3m2fn_f32)
#error "deferred arbitrary FP conversion builtin is unexpectedly available"
#endif

float convert_scalar(uchar src) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src);
}

_Float16 convert_f16(uchar src) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(src);
}

__bf16 convert_bf16(uchar src) {
  return __builtin_elementwise_convert_from_f8e5m2_bf16(src);
}

float4 convert_vector(uchar4 src) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src);
}

// f16 denotes _Float16 even in OpenCL; 'half' comes from the normal
// conversion rules.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half convert_to_half(uchar src) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(src);
}

half4 convert_to_half_vector(uchar4 src) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(src);
}
