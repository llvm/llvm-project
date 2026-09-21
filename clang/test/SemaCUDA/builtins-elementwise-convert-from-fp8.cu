// RUN: %clang_cc1 -x cuda -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -fsyntax-only -verify=supported %s
// RUN: %clang_cc1 -x cuda -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -fsyntax-only -verify=supported %s
// RUN: %clang_cc1 -x hip -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -fsyntax-only -verify=supported %s
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -fsyntax-only -verify=supported %s
// RUN: %clang_cc1 -x cuda -triple i386-unknown-linux-gnu -aux-triple x86_64-unknown-linux-gnu -target-feature -sse2 -fcuda-is-device -DUNSUPPORTED_TARGET -fsyntax-only -verify=unsupported %s
// RUN: %clang_cc1 -x hip -triple i386-unknown-linux-gnu -aux-triple x86_64-unknown-linux-gnu -target-feature -sse2 -fcuda-is-device -DUNSUPPORTED_TARGET -fsyntax-only -verify=unsupported %s
// supported-no-diagnostics

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef unsigned char uchar3 __attribute__((ext_vector_type(3)));

#ifdef UNSUPPORTED_TARGET

// Unsupported result types in host-only or unused inline host/device functions
// must not be diagnosed during device compilation.
__host__ void host_only(unsigned char bits, uchar3 packed) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits);
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(packed);
}

__host__ __device__ inline void unused(unsigned char bits, uchar3 packed) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits);
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(packed);
}

__host__ __device__ inline void used(unsigned char bits, uchar3 packed) {
  // CUDA permits spelling _Float16 even when one compilation target does not
  // support it, and unevaluated operands do not require device type support.
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e5m2_f16(bits)),
                          _Float16), "");
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits); // unsupported-error {{expression requires 16 bit size '_Float16' type support, but target 'i386-unknown-linux-gnu' does not support it}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(packed); // unsupported-error {{expression requires 16 bit size '_Float16' type support, but target 'i386-unknown-linux-gnu' does not support it}}
}

__global__ void kernel(unsigned char bits, uchar3 packed) {
  used(bits, packed); // unsupported-note {{called by 'kernel'}}
}

#else

typedef _Float16 float16_3 __attribute__((ext_vector_type(3)));
typedef __bf16 bfloat16_3 __attribute__((ext_vector_type(3)));
typedef float float3 __attribute__((ext_vector_type(3)));

__host__ __device__ void results(unsigned char bits, uchar3 packed) {
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e5m2_f16(bits)),
                          _Float16), "");
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits)),
                          __bf16), "");
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits)),
                          float), "");
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e5m2_f16(packed)),
                          float16_3), "");
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e4m3fn_bf16(packed)),
                          bfloat16_3), "");
  static_assert(__is_same(decltype(__builtin_elementwise_convert_from_f8e5m3fnu_f32(packed)),
                          float3), "");

  (void)__builtin_elementwise_convert_from_f8e5m2_f16(bits);
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(packed);
  (void)__builtin_elementwise_convert_from_f8e4m3fn_bf16(packed);
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(packed);
}

__host__ void host(unsigned char bits, uchar3 packed) {
  results(bits, packed);
}

__global__ void kernel(unsigned char bits, uchar3 packed) {
  results(bits, packed);
}

#endif
