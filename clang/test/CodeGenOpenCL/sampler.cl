// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -cl-std=CL2.0 -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -cl-std=clc++ -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck %s
//
// This test covers 5 cases of sampler initialzation:
//   1. function argument passing
//      1a. argument is a file-scope variable
//      1b. argument is a function-scope variable
//      1c. argument is one of caller function's parameters
//   2. variable initialization
//      2a. initializing a file-scope variable with constant addr space qualifier
//      2b. initializing a function-scope variable
//      2c. initializing a file-scope variable with const qualifier

#define CLK_ADDRESS_CLAMP_TO_EDGE       2
#define CLK_NORMALIZED_COORDS_TRUE      1
#define CLK_FILTER_NEAREST              0x10
#define CLK_FILTER_LINEAR               0x20

// CHECK: %opencl.sampler_t = type opaque

// Case 2a
constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
// CHECK-NOT: glb_smp

// Case 2c
const sampler_t glb_smp_const = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
// CHECK-NOT: glb_smp_const

int get_sampler_initializer(void);

void fnc4smp(sampler_t s) {}
// CHECK: define{{.*}} spir_func void [[FUNCNAME:@.*fnc4smp.*]](%opencl.sampler_t addrspace(2)* %

kernel void foo(sampler_t smp_par) {
  // CHECK-LABEL: define{{.*}} spir_kernel void @foo(%opencl.sampler_t addrspace(2)* %smp_par)
  // CHECK: [[smp_par_ptr:%[A-Za-z0-9_\.]+]] = alloca %opencl.sampler_t addrspace(2)*

  // Case 2b
  sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
  // CHECK: [[smp_ptr:%[A-Za-z0-9_\.]+]] = alloca %opencl.sampler_t addrspace(2)*
  // CHECK: [[SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 19)
  // CHECK: store %opencl.sampler_t addrspace(2)* [[SAMP]], %opencl.sampler_t addrspace(2)** [[smp_ptr]]

  // Case 1b
  fnc4smp(smp);
  // CHECK-NOT: call %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 19)
  // CHECK: [[SAMP:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[smp_ptr]]
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  // Case 1b
  fnc4smp(smp);
  // CHECK-NOT: call %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 19)
  // CHECK: [[SAMP:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[smp_ptr]]
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  // Case 1a/2a
  fnc4smp(glb_smp);
  // CHECK: [[SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 35)
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  // Case 1a/2c
  fnc4smp(glb_smp_const);
  // CHECK: [[SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 35)
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  // Case 1c
  fnc4smp(smp_par);
  // CHECK: [[SAMP:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[smp_par_ptr]]
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  fnc4smp(5);
  // CHECK: [[SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 5)
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  const sampler_t const_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  fnc4smp(const_smp);
   // CHECK: [[CONST_SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 35)
  // CHECK: store %opencl.sampler_t addrspace(2)* [[CONST_SAMP]], %opencl.sampler_t addrspace(2)** [[CONST_SMP_PTR:%[a-zA-Z0-9]+]]
  fnc4smp(const_smp);
  // CHECK: [[SAMP:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[CONST_SMP_PTR]]
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  constant sampler_t constant_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  fnc4smp(constant_smp);
  // CHECK: [[SAMP:%[0-9]+]] = call spir_func %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 35)
  // CHECK: call spir_func void [[FUNCNAME]](%opencl.sampler_t addrspace(2)* [[SAMP]])

  // TODO: enable sampler initialization with non-constant integer.
  //const sampler_t const_smp_func_init = get_sampler_initializer();
}
