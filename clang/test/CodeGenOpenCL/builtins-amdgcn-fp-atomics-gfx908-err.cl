// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx908 \
// RUN:   -verify -S -o - %s

typedef half __attribute__((ext_vector_type(2))) half2;

void test_global_add_2f16(__global half2 *addrh2, half2 xh2,
                          __global float *addrf, float xf,
                          __global double *addr, double x) {
  half2 *half_rtn;
  float *fp_rtn;
  double *rtn;
  *half_rtn = __builtin_amdgcn_global_atomic_fadd_v2f16(addrh2, xh2); // expected-error{{'__builtin_amdgcn_global_atomic_fadd_v2f16' needs target feature gfx90a-insts}}
  *fp_rtn = __builtin_amdgcn_global_atomic_fadd_f32(addr, x); // expected-error{{'__builtin_amdgcn_global_atomic_fadd_f32' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_global_atomic_fadd_f64(addr, x); // expected-error{{'__builtin_amdgcn_global_atomic_fadd_f64' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_global_atomic_fmax_f64(addr, x); // expected-error{{'__builtin_amdgcn_global_atomic_fmax_f64' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_global_atomic_fmin_f64(addr, x); // expected-error{{'__builtin_amdgcn_global_atomic_fmin_f64' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_flat_atomic_fadd_f64(addr, x); // expected-error{{'__builtin_amdgcn_flat_atomic_fadd_f64' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_flat_atomic_fmin_f64(addr, x); // expected-error{{'__builtin_amdgcn_flat_atomic_fmin_f64' needs target feature gfx90a-insts}}
  *rtn = __builtin_amdgcn_flat_atomic_fmax_f64(addr, x); // expected-error{{'__builtin_amdgcn_flat_atomic_fmax_f64' needs target feature gfx90a-insts}}
}
