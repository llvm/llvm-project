// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -fsyntax-only -verify=dev %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=host %s

// host-no-diagnostics

#include "Inputs/cuda.h"

int global_host_var;
__device__ int global_dev_var;
__constant__ int global_constant_var;
__shared__ int global_shared_var;
constexpr int global_constexpr_var = 1;
const int global_const_var = 1;

template<typename F>
__global__ void kernel(F f) { f(); } // dev-note2 {{called by 'kernel<(lambda}}

__device__ void dev_fun(int *out) {
  int &ref_host_var = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
}

__global__ void global_fun(int *out) {
  int &ref_host_var = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __global__ function}}
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __global__ function}}
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
}

__host__ __device__ void host_dev_fun(int *out) {
  int &ref_host_var = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __host__ __device__ function}}
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __host__ __device__ function}}
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
}

inline __host__ __device__ void inline_host_dev_fun(int *out) {
  int &ref_host_var = global_host_var;
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var;
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
}

void dev_lambda_capture_by_ref(int *out) {
  int &ref_host_var = global_host_var;
  kernel<<<1,1>>>([&]() {
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __host__ __device__ function}}
                          // dev-error@-1 {{capture host variable 'out' by reference in device or host device lambda function}}
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var; // dev-error {{capture host variable 'ref_host_var' by reference in device or host device lambda function}}
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
  });
}

void dev_lambda_capture_by_copy(int *out) {
  int &ref_host_var = global_host_var;
  kernel<<<1,1>>>([=]() {
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  const int &ref_constexpr_var = global_constexpr_var;
  const int &ref_const_var = global_const_var;

  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __host__ __device__ function}}
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;
  *out = global_constexpr_var;
  *out = global_const_var;

  *out = ref_host_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;
  });
}

// Texture references are special. As far as C++ is concerned they are host
// variables that are referenced from device code. However, they are handled
// very differently by the compiler under the hood and such references are
// allowed. Compiler should produce no warning here, but it should diagnose the
// same case without the device_builtin_texture_type attribute.
template <class, int = 1, int = 1>
struct __attribute__((device_builtin_texture_type)) texture {
  static texture<int> ref;
  __device__ int c() {
    auto &x = ref;
  }
};

template <class, int = 1, int = 1>
struct  not_a_texture {
  static not_a_texture<int> ref;
  __device__ int c() {
    auto &x = ref; // dev-error {{reference to __host__ variable 'ref' in __device__ function}}
  }
};
