// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck --check-prefixes=COMMON,WITH %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - -DNO_CONSTANTS %s | FileCheck --check-prefixes=COMMON,EMPTY %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DCONSTANT_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DRESERVED_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DDEVICE_CONST_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DPRAGMA_CONSTANT_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DPRAGMA_RESERVED_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DPRAGMA_DEVICE_CONST_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DPRECEDENCE_NO_DIAGNOSTICS -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DPRECEDENCE_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DTEMPLATE_DEPENDENT_NO_INSTANTIATION -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fsyntax-only -DTEMPLATE_DEPENDENT_SECTION_ERROR -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o /dev/null -DBOUNDARY_SYMBOL_COLLISION_ERROR -verify %s

#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))

#ifdef CONSTANT_SECTION_ERROR
__constant__ int constant_wrong_section __attribute__((section("user_constant"))) = 1;
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
#elif defined(RESERVED_SECTION_ERROR)
__device__ int device_wrong_section __attribute__((section("__amdgpu_constant"))) = 1;
// expected-error@-1 {{section '__amdgpu_constant' is reserved for AMDGPU constant address space variables}}
#elif defined(DEVICE_CONST_SECTION_ERROR)
__device__ const int device_const_wrong_section __attribute__((section("user_constant"))) = 1;
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
#elif defined(PRAGMA_CONSTANT_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "user_constant"
__constant__ int constant_wrong_pragma = 1;
#pragma clang section data = ""
#elif defined(PRAGMA_RESERVED_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "__amdgpu_constant"
__device__ int device_wrong_pragma = 1;
#pragma clang section data = ""
#elif defined(PRAGMA_DEVICE_CONST_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "user_constant"
__device__ const int device_const_wrong_pragma = 1;
#pragma clang section data = ""
#elif defined(PRECEDENCE_NO_DIAGNOSTICS)
// expected-no-diagnostics
#pragma clang section data = "user_constant"
__constant__ int explicit_constant_section_precedence
    __attribute__((section("__amdgpu_constant"))) = 1;
#pragma clang section data = ""
#pragma clang section data = "__amdgpu_constant"
__device__ int explicit_device_section_precedence
    __attribute__((section("user_data"))) = 1;
#pragma clang section data = ""
#elif defined(PRECEDENCE_SECTION_ERROR)
#pragma clang section data = "__amdgpu_constant"
__constant__ int explicit_wrong_constant_section_precedence
    __attribute__((section("user_constant"))) = 1;
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
#pragma clang section data = ""
#elif defined(TEMPLATE_DEPENDENT_NO_INSTANTIATION)
// expected-no-diagnostics
template <class T>
__device__ const T dependent_wrong_section __attribute__((section("user_constant"))) = T(1);
#elif defined(TEMPLATE_DEPENDENT_SECTION_ERROR)
template <class T>
__device__ const T dependent_wrong_section __attribute__((section("user_constant"))) = T(1);
// expected-error@-1 {{AMDGPU constant address space variables must use section '__amdgpu_constant'}}
__device__ int use_dependent_wrong_section() {
  return dependent_wrong_section<int>;
  // expected-note@-1 {{in instantiation of variable template specialization 'dependent_wrong_section<int>' requested here}}
}
#elif defined(BOUNDARY_SYMBOL_COLLISION_ERROR)
extern "C" __device__ int __start___amdgpu_constant;
__device__ bool boundary_symbol_collision(const void *ptr) {
  (void)&__start___amdgpu_constant;
  return __builtin_amdgcn_is_constant(ptr);
  // expected-error@-1 {{AMDGPU constant section boundary symbol '__start___amdgpu_constant' conflicts with existing symbol}}
}
#elif !defined(NO_CONSTANTS)
// WITH-DAG: @constant_var = addrspace(4) externally_initialized constant i32 1, section "__amdgpu_constant"
__constant__ int constant_var = 1;
#endif

// COMMON-DAG: @device_var = addrspace(1) externally_initialized global i32 0
__device__ int device_var;

// COMMON-DAG: @device_const_var = addrspace(4) constant i32 2, section "__amdgpu_constant"
extern "C" __device__ const int device_const_var = 2;

// COMMON-DAG: @__clang_section_marker___amdgpu_constant = internal unnamed_addr addrspace(4) constant [0 x i8] zeroinitializer, section "__amdgpu_constant"
// COMMON-DAG: @__start___amdgpu_constant = external addrspace(4) constant i8
// COMMON-DAG: @__stop___amdgpu_constant = external addrspace(4) constant i8
// COMMON-DAG: @llvm.compiler.used = {{.*}}@__clang_section_marker___amdgpu_constant{{.*}}@__hip_cuid_{{.*}}, section "llvm.metadata"
// EMPTY-NOT: @constant_var =

// COMMON-LABEL: define{{.*}} noundef zeroext i1 @_Z11is_constantPKv(
// COMMON: [[PTR:%.*]] = load ptr, ptr {{.*}}, align 8
// COMMON: [[PTR_ADDR:%.*]] = ptrtoaddr ptr [[PTR]] to i64
// COMMON: [[OFFSET:%.*]] = sub i64 [[PTR_ADDR]], ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__start___amdgpu_constant {{.*}}ptr) to i64)
// COMMON: [[CMP:%.*]] = icmp ult i64 [[OFFSET]], sub (i64 ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__stop___amdgpu_constant {{.*}}ptr) to i64), i64 ptrtoaddr (ptr addrspacecast (ptr addrspace(4) @__start___amdgpu_constant {{.*}}ptr) to i64))
// COMMON: ret i1 [[CMP]]
__device__ bool is_constant(const void *ptr) {
  return __builtin_amdgcn_is_constant(ptr);
}
