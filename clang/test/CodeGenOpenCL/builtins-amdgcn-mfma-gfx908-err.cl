// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx908 \
// RUN:   -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp64:enable

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v32f  __attribute__((ext_vector_type(32)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef half   v32h  __attribute__((ext_vector_type(32)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));
typedef int    v32i  __attribute__((ext_vector_type(32)));
typedef short  v2s   __attribute__((ext_vector_type(2)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef short  v16s  __attribute__((ext_vector_type(16)));
typedef short  v32s  __attribute__((ext_vector_type(32)));
typedef double v4d   __attribute__((ext_vector_type(4)));

void test_mfma_f32_16x16x4bf16_1k(global v16f* out, global v4f* out1,
                                  global v4d* out2, global double* out3, v4s a,
                                  v4s b, v16f c, v4f e, double f, double g,
                                  v4d h)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a, b, c, 0, 0, 0);   // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' needs target feature gfx90a-insts}}
  *out1 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, e, 0, 0, 0);    // expected-error{{'__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' needs target feature gfx90a-insts}}
  *out = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, 0);   // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' needs target feature gfx90a-insts}}
  *out1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, e, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' needs target feature gfx90a-insts}}
  *out2 = __builtin_amdgcn_mfma_f64_16x16x4f64(f, g, h, 0, 0, 0);      // expected-error{{'__builtin_amdgcn_mfma_f64_16x16x4f64' needs target feature gfx90a-insts}}
  *out3 = __builtin_amdgcn_mfma_f64_4x4x4f64(f, g, g, 0, 0, 0);        // expected-error{{'__builtin_amdgcn_mfma_f64_4x4x4f64' needs target feature gfx90a-insts}}
}
