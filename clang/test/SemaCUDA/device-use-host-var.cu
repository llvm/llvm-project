// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -fsyntax-only -verify=dev %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=host %s

// host-no-diagnostics

#include "Inputs/cuda.h"

int func();

struct A {
  int x;
  static int host_var;
};

int A::host_var; // dev-note {{host variable declared here}}

namespace X {
  int host_var; // dev-note {{host variable declared here}}
}

// struct with non-empty ctor.
struct B1 {
  int x;
  B1() { x = 1; }
};

// struct with non-empty dtor.
struct B2 {
  int x;
  B2() {}
  ~B2() { x = 0; }
};

static int static_host_var; // dev-note {{host variable declared here}}

__device__ int global_dev_var;
__constant__ int global_constant_var;
__shared__ int global_shared_var;

int global_host_var; // dev-note 8{{host variable declared here}}
const int global_const_var = 1;
constexpr int global_constexpr_var = 1;

int global_host_array[2] = {1, 2}; // dev-note {{host variable declared here}}
const int global_const_array[2] = {1, 2};
constexpr int global_constexpr_array[2] = {1, 2};

A global_host_struct_var{1}; // dev-note 2{{host variable declared here}}
const A global_const_struct_var{1};
constexpr A global_constexpr_struct_var{1};

// Check const host var initialized with non-empty ctor is not allowed in
// device function.
const B1 b1; // dev-note {{const variable cannot be emitted on device side due to dynamic initialization}}

// Check const host var having non-empty dtor is not allowed in device function.
const B2 b2; // dev-note {{const variable cannot be emitted on device side due to dynamic initialization}}

// Check const host var initialized by non-constant initializer is not allowed
// in device function.
const int b3 = func(); // dev-note {{const variable cannot be emitted on device side due to dynamic initialization}}

template<typename F>
__global__ void kernel(F f) { f(); } // dev-note2 {{called by 'kernel<(lambda}}

__device__ void dev_fun(int *out) {
  // Check access device variables are allowed.
  int &ref_dev_var = global_dev_var;
  int &ref_constant_var = global_constant_var;
  int &ref_shared_var = global_shared_var;
  *out = ref_dev_var;
  *out = ref_constant_var;
  *out = ref_shared_var;
  *out = global_dev_var;
  *out = global_constant_var;
  *out = global_shared_var;

  // Check access of non-const host variables are not allowed.
  *out = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}
  *out = global_const_var;
  *out = global_constexpr_var;
  *out = b1.x; // dev-error {{reference to __host__ variable 'b1' in __device__ function}}
  *out = b2.x; // dev-error {{reference to __host__ variable 'b2' in __device__ function}}
  *out = b3; // dev-error {{reference to __host__ variable 'b3' in __device__ function}}
  global_host_var = 1; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}

  // Check reference of non-constexpr host variables are not allowed.
  int &ref_host_var = global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}
  const int &ref_const_var = global_const_var;
  const int &ref_constexpr_var = global_constexpr_var;
  *out = ref_host_var;
  *out = ref_constexpr_var;
  *out = ref_const_var;

  // Check access member of non-constexpr struct type host variable is not allowed.
  *out = global_host_struct_var.x; // dev-error {{reference to __host__ variable 'global_host_struct_var' in __device__ function}}
  *out = global_const_struct_var.x;
  *out = global_constexpr_struct_var.x;
  global_host_struct_var.x = 1; // dev-error {{reference to __host__ variable 'global_host_struct_var' in __device__ function}}

  // Check address taking of non-constexpr host variables is not allowed.
  int *p = &global_host_var; // dev-error {{reference to __host__ variable 'global_host_var' in __device__ function}}
  const int *cp = &global_const_var;
  const int *cp2 = &global_constexpr_var;

  // Check access elements of non-constexpr host array is not allowed.
  *out = global_host_array[1]; // dev-error {{reference to __host__ variable 'global_host_array' in __device__ function}}
  *out = global_const_array[1];
  *out = global_constexpr_array[1];

  // Check ODR-use of host variables in namespace is not allowed.
  *out = X::host_var; // dev-error {{reference to __host__ variable 'host_var' in __device__ function}}

  // Check ODR-use of static host varables in class or file scope is not allowed.
  *out = A::host_var; // dev-error {{reference to __host__ variable 'host_var' in __device__ function}}
  *out = static_host_var; // dev-error {{reference to __host__ variable 'static_host_var' in __device__ function}}

  // Check function-scope static variable is allowed.
  static int static_var;
  *out = static_var;

  // Check non-ODR use of host varirables are allowed.
  *out = sizeof(global_host_var);
  *out = sizeof(global_host_struct_var.x);
  decltype(global_host_var) var1;
  decltype(global_host_struct_var.x) var2;
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
  __device__ void c() {
    auto &x = ref;
  }
};

template <class, int = 1, int = 1>
struct  not_a_texture {
  static not_a_texture<int> ref;
  __device__ void c() {
    auto &x = ref; // dev-error {{reference to __host__ variable 'ref' in __device__ function}}
  }
};

template<>
not_a_texture<int> not_a_texture<int>::ref; // dev-note {{host variable declared here}}

__device__ void test_not_a_texture() {
  not_a_texture<int> inst;
  inst.c(); // dev-note {{in instantiation of member function 'not_a_texture<int, 1, 1>::c' requested here}}
}

// Test static variable in host function used by device function.
void test_static_var_host() {
  for (int i = 0; i < 10; i++) {
    static int x; // dev-note {{host variable declared here}}
    struct A {
      __device__ int f() {
        return x; // dev-error{{reference to __host__ variable 'x' in __device__ function}}
      }
    };
  }
}

// Test static variable in device function used by device function.
__device__ void test_static_var_device() {
  for (int i = 0; i < 10; i++) {
    static int x;
    int y = x;
    struct A {
      __device__ int f() {
        return x;
      }
    };
  }
}
