// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -emit-llvm -o - -triple spirv64-unknown-unknown %s | FileCheck %s
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -S -o - -triple spirv64-unknown-unknown -mllvm --spirv-ext=+SPV_INTEL_function_pointers %s | FileCheck %s --check-prefix=SPIRV

// Check that local sizes are passed as a pointer to the first array element
// and that the SPIR-V backend lowers this form.

typedef struct {int a;} ndrange_t;

kernel void test(queue_t q) {
  ndrange_t nd;
// CHECK-LABEL: define {{.*}} void @__clang_ocl_kern_imp_test(
// CHECK: %[[SIZES:.*]] = alloca [2 x i64]
// CHECK: %[[ELEM:.*]] = getelementptr [2 x i64], ptr %[[SIZES]], i32 0, i32 0
// CHECK: call spir_func i32 @__enqueue_kernel_varargs({{.*}}, i32 2, ptr %[[ELEM]])

// SPIRV-DAG: %[[#INT32:]] = OpTypeInt 32 0
// SPIRV-DAG: %[[#ZERO:]] = OpConstantNull %[[#INT32]]
// SPIRV: %[[#ELEM:]] = OpPtrAccessChain %[[#]] %[[#]] %[[#ZERO]] %[[#ZERO]]
// SPIRV: %[[#SIZE0:]] = OpPtrAccessChain %[[#]] %[[#ELEM]] %[[#ZERO]]{{$}}
// SPIRV-NEXT: %[[#SIZE1:]] = OpPtrAccessChain %[[#]] %[[#ELEM]] %[[#]]{{$}}
// SPIRV-NEXT: OpEnqueueKernel {{.*}} %[[#SIZE0]] %[[#SIZE1]]{{$}}
  enqueue_kernel(q, 0, nd, ^(local void *p1, local void *p2){}, 101u, 102u);
}
