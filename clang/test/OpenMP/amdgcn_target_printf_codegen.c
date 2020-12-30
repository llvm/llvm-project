// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-is-device -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK
// expected-no-diagnostics

extern int printf(const char *, ...);

// CHECK-DAG: [[CHECK_ZERO_ARG_TY:%[a-zA-Z0-9_.]+]] = type { i32, i32, i32, i32 }
// CHECK-DAG: [[CHECK_ZERO_ARG_STR:@[a-zA-Z0-9_.]+]] = private unnamed_addr addrspace(4) constant [13 x i8] c"Hello, world\00", align 1
// CHECK-DAG: define weak amdgpu_kernel void @__omp_offloading{{.+}}CheckZeroArg{{.+}}
int CheckZeroArg() {
  // size passed to printf_alloc (Hello, world + \0) 13 bytes + 4 bytes + 4 bytes + 4 bytes + 4 bytes = 29

  // CHECK: [[PTR0:%[a-zA-Z0-9_]+]] = call i8* @printf_allocate(i32 29)
  // CHECK: [[ARGS_CAST:%[a-zA-Z0-9_]+]] = addrspacecast i8* [[PTR0]] to [[CHECK_ZERO_ARG_TY]] addrspace(1)*
  #pragma omp target
  {
    // CHECK: [[SIZE:%[0-9]+]] = getelementptr inbounds [[CHECK_ZERO_ARG_TY]], [[CHECK_ZERO_ARG_TY]] addrspace(1)* [[ARGS_CAST]], i32 0, i32 0
    // CHECK: store i32 16, i32 addrspace(1)* [[SIZE]]

    // CHECK: [[NUM_ARG:%[0-9]+]] = getelementptr inbounds [[CHECK_ZERO_ARG_TY]], [[CHECK_ZERO_ARG_TY]] addrspace(1)* [[ARGS_CAST]], i32 0, i32 1
    // CHECK: store i32 1, i32 addrspace(1)* [[NUM_ARG]]

    // CHECK: [[PTR1:%[0-9]+]] = getelementptr inbounds [[CHECK_ZERO_ARG_TY]], [[CHECK_ZERO_ARG_TY]] addrspace(1)* [[ARGS_CAST]], i32 0, i32 2
    // CHECK: store i32 {{[0-9]+}}, i32 addrspace(1)* [[PTR1]]

    // CHECK: [[PTR2:%[0-9]+]] = getelementptr inbounds [[CHECK_ZERO_ARG_TY]], [[CHECK_ZERO_ARG_TY]] addrspace(1)* [[ARGS_CAST]], i32 0, i32
    // CHECK: store i32 13, i32 addrspace(1)* [[PTR2]], align 4
    // CHECK: [[PTR3:%[0-9]+]] = bitcast [[CHECK_ZERO_ARG_TY]] addrspace(1)* [[ARGS_CAST]] to i8 addrspace(1)*
    // CHECK: [[PTR4:%[0-9]+]] = getelementptr inbounds i8, i8 addrspace(1)* [[PTR3]], i64 16
    // CHECK: call void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* align 1 [[PTR4]], i8* align 1 addrspacecast (i8 addrspace(4)* getelementptr inbounds ([13 x i8], [13 x i8] addrspace(4)* [[CHECK_ZERO_ARG_STR]], i32 0, i32 0) to i8*), i64 13, i1 false)
    // CHECK: [[PTR5:%[0-9]+]] = getelementptr inbounds i8, i8 addrspace(1)* [[PTR4]], i64 13
    // CHECK: [[PTR6:%[0-9]+]] = call i32 @printf_execute(i8* [[PTR0]], i32 29)
    printf("Hello, world");
  }

  return 0;
}


