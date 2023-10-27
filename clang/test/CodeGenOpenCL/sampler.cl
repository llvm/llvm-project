// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-SPIR %s
// RUN: %clang_cc1 %s -cl-std=clc++ -emit-llvm -triple spir-unknown-unknown -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-SPIR %s
// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-unknown-linux-gnu -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-X86 %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -triple x86_64-unknown-linux-gnu -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-X86 %s
// RUN: %clang_cc1 %s -cl-std=clc++ -emit-llvm -triple x86_64-unknown-linux-gnu -o - -O0 | FileCheck --check-prefixes=CHECK-COMMON,CHECK-X86 %s
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

// Case 2a
constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
// CHECK-COMMON-NOT: glb_smp

// Case 2c
const sampler_t glb_smp_const = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
// CHECK-COMMON-NOT: glb_smp_const

int get_sampler_initializer(void);

void fnc4smp(sampler_t s) {}
// CHECK-SPIR: define{{.*}} spir_func void [[FUNCNAME:@.*fnc4smp.*]](target("spirv.Sampler") %
// CHECK-X86: define{{.*}} void [[FUNCNAME:@.*fnc4smp.*]](ptr %

kernel void foo(sampler_t smp_par) {
  // CHECK-SPIR-LABEL: define{{.*}} spir_kernel void @foo(target("spirv.Sampler") %smp_par)
  // CHECK-SPIR: [[smp_par_ptr:%[A-Za-z0-9_\.]+]] = alloca target("spirv.Sampler")
  // CHECK-X86-LABEL: define{{.*}} spir_kernel void @foo(ptr %smp_par)
  // CHECK-X86: [[smp_par_ptr:%[A-Za-z0-9_\.]+]] = alloca ptr

  // Case 2b
  sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
  // CHECK-SPIR: [[smp_ptr:%[A-Za-z0-9_\.]+]] = alloca target("spirv.Sampler")
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 19)
  // CHECK-SPIR: store target("spirv.Sampler") [[SAMP]], ptr [[smp_ptr]]
  // CHECK-X86: [[smp_ptr:%[A-Za-z0-9_\.]+]] = alloca ptr
  // CHECK-X86: [[SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 19)
  // CHECK-X86: store ptr [[SAMP]], ptr [[smp_ptr]]

  // Case 1b
  fnc4smp(smp);
  // CHECK-SPIR-NOT: call target("spirv.Sampler") @__translate_sampler_initializer(i32 19)
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = load target("spirv.Sampler"), ptr [[smp_ptr]]
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86-NOT: call ptr @__translate_sampler_initializer(i32 19)
  // CHECK-X86: [[SAMP:%[0-9]+]] = load ptr, ptr [[smp_ptr]]
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  // Case 1b
  fnc4smp(smp);
  // CHECK-SPIR-NOT: call target("spirv.Sampler") @__translate_sampler_initializer(i32 19)
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = load target("spirv.Sampler"), ptr [[smp_ptr]]
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86-NOT: call ptr @__translate_sampler_initializer(i32 19)
  // CHECK-X86: [[SAMP:%[0-9]+]] = load ptr, ptr [[smp_ptr]]
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  // Case 1a/2a
  fnc4smp(glb_smp);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 35)
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 35)
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  // Case 1a/2c
  fnc4smp(glb_smp_const);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 35)
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 35)
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  // Case 1c
  fnc4smp(smp_par);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = load target("spirv.Sampler"), ptr [[smp_par_ptr]]
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = load ptr, ptr [[smp_par_ptr]]
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  fnc4smp(5);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 5)
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 5)
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  const sampler_t const_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  fnc4smp(const_smp);
  // CHECK-SPIR: [[CONST_SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 35)
  // CHECK-SPIR: store target("spirv.Sampler") [[CONST_SAMP]], ptr [[CONST_SMP_PTR:%[a-zA-Z0-9]+]]
  // CHECK-X86: [[CONST_SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 35)
  // CHECK-X86: store ptr [[CONST_SAMP]], ptr [[CONST_SMP_PTR:%[a-zA-Z0-9]+]]
  fnc4smp(const_smp);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = load target("spirv.Sampler"), ptr [[CONST_SMP_PTR]]
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = load ptr, ptr [[CONST_SMP_PTR]]
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  constant sampler_t constant_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  fnc4smp(constant_smp);
  // CHECK-SPIR: [[SAMP:%[0-9]+]] = call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 35)
  // CHECK-SPIR: call spir_func void [[FUNCNAME]](target("spirv.Sampler") [[SAMP]])
  // CHECK-X86: [[SAMP:%[0-9]+]] = call ptr @__translate_sampler_initializer(i32 35)
  // CHECK-X86: call void [[FUNCNAME]](ptr [[SAMP]])

  // TODO: enable sampler initialization with non-constant integer.
  //const sampler_t const_smp_func_init = get_sampler_initializer();
}
