// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DCONSTANT_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DPRAGMA_CONSTANT_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DRESERVED_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DPRAGMA_RESERVED_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DGLOBAL_CONST_RESERVED_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DPRECEDENCE_CONSTANT_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DPRECEDENCE_GLOBAL_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -DPRECEDENCE_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -emit-llvm -o /dev/null -DBOUNDARY_SYMBOL_COLLISION_ERROR -verify %s

#ifdef CONSTANT_SECTION_ERROR
__constant int constant_wrong_section __attribute__((section("user_constant"))) = 1;
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
#elif defined(PRAGMA_CONSTANT_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "user_constant"
__constant int constant_wrong_pragma = 1;
#pragma clang section data = ""
#elif defined(RESERVED_SECTION_ERROR)
__global int global_wrong_section __attribute__((section("__amdgpu_constant"))) = 1;
// expected-error@-1 {{section '__amdgpu_constant' is reserved for AMDGPU constant address space variables}}
#elif defined(PRAGMA_RESERVED_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "__amdgpu_constant"
__global int global_wrong_pragma = 1;
#pragma clang section data = ""
#elif defined(GLOBAL_CONST_RESERVED_SECTION_ERROR)
__global const int global_const_wrong_section __attribute__((section("__amdgpu_constant"))) = 1;
// expected-error@-1 {{section '__amdgpu_constant' is reserved for AMDGPU constant address space variables}}
#elif defined(PRECEDENCE_CONSTANT_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "user_constant"
__constant int explicit_constant_section_precedence
    __attribute__((section("__amdgpu_constant"))) = 1;
#pragma clang section data = ""
#elif defined(PRECEDENCE_GLOBAL_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "__amdgpu_constant"
__global int explicit_global_section_precedence
    __attribute__((section("user_global"))) = 1;
#pragma clang section data = ""
#elif defined(PRECEDENCE_SECTION_ERROR)
#pragma clang section data = "__amdgpu_constant"
__constant int explicit_wrong_constant_section_precedence
    __attribute__((section("user_constant"))) = 1;
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
#pragma clang section data = ""
#elif defined(BOUNDARY_SYMBOL_COLLISION_ERROR)
__global int __start___amdgpu_constant;
bool boundary_symbol_collision(const void *ptr) {
  (void)&__start___amdgpu_constant;
  return __builtin_amdgcn_is_constant(ptr);
  // expected-error@-1 {{AMDGPU constant section boundary symbol '__start___amdgpu_constant' conflicts with existing symbol}}
}
#else
// CHECK-DAG: @constant_var = {{.*}}addrspace(4) constant i32 1, section "__amdgpu_constant"
__constant int constant_var = 1;

// CHECK-DAG: @global_var = {{.*}}addrspace(1) global i32 0
__global int global_var;

// CHECK-DAG: @global_const_var = {{.*}}addrspace(1) constant i32 2, align 4{{$}}
__global const int global_const_var = 2;

// CHECK-DAG: @__clang_section_marker___amdgpu_constant = internal unnamed_addr addrspace(4) constant [0 x i8] zeroinitializer, section "__amdgpu_constant"
// CHECK-DAG: @__start___amdgpu_constant = external addrspace(4) constant i8
// CHECK-DAG: @__stop___amdgpu_constant = external addrspace(4) constant i8
// CHECK-DAG: @llvm.compiler.used = {{.*}}@__clang_section_marker___amdgpu_constant{{.*}}, section "llvm.metadata"
#endif

// CHECK-LABEL: define{{.*}} zeroext i1 @is_constant(
// CHECK-SAME: ptr {{.*}}noundef {{.*}}[[PTR:%[[:alnum:]_.]+]]
// CHECK: [[PTR_ADDR:%.*]] = ptrtoaddr ptr [[PTR]] to i64
// CHECK: [[OFFSET:%.*]] = sub i64 [[PTR_ADDR]], ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__start___amdgpu_constant {{.*}}ptr) to i64)
// CHECK: [[CMP:%.*]] = icmp ult i64 [[OFFSET]], sub (i64 ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__stop___amdgpu_constant {{.*}}ptr) to i64), i64 ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__start___amdgpu_constant {{.*}}ptr) to i64))
// CHECK: ret i1 [[CMP]]
bool is_constant(const void *ptr) {
  return __builtin_amdgcn_is_constant(ptr);
}
