// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device \
// RUN:   -mcode-object-version=4 -fno-offload-uniform-block -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=COV4 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -mcode-object-version=4 -fno-offload-uniform-block -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=COV4 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device \
// RUN:   -mcode-object-version=5 -fno-offload-uniform-block -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=COV5 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -mcode-object-version=5 -fno-offload-uniform-block -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=COV5 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device \
// RUN:   -mcode-object-version=none -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=NONE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -mcode-object-version=none -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=NONE %s

#define __device__ __attribute__((device))

// NONE: @__oclc_ABI_version = external addrspace(4) global i32

// COV4-LABEL: @_Z4testv
// COV4: [[DP:%.*]] = call{{.*}} ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// COV4: [[GEP:%.*]] = getelementptr{{.*}} i8, ptr addrspace(4) [[DP]], i64 6
// COV4: [[LD:%.*]] = load i16, ptr addrspace(4) [[GEP]], align 2
// COV4: [[GS:%.*]] = zext i16 [[LD]] to i32
// COV4: [[ID:%.*]] = call i32 @llvm.amdgcn.workgroup.id.y()
// COV4: [[GGEP:%.*]] = getelementptr{{.*}} i8, ptr addrspace(4) [[DP]], i64 16
// COV4: [[GRID:%.*]] = load i32, ptr addrspace(4) [[GGEP]], align 4
// COV4: [[MUL:%.*]] = mul i32 [[ID]], [[GS]]
// COV4: [[REM:%.*]] = sub i32 [[GRID]], [[MUL]]
// COV4: [[CMP:%.*]] = icmp ult i32 [[REM]], [[GS]]
// COV4: select i1 [[CMP]], i32 [[REM]], i32 [[GS]]

// COV5-LABEL: @_Z4testv
// COV5: [[IP:%.*]] = call{{.*}} ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// COV5: [[BGEP:%.*]] = getelementptr{{.*}} i8, ptr addrspace(4) [[IP]], i64 4
// COV5: [[BC:%.*]] = load i32, ptr addrspace(4) [[BGEP]], align 4
// COV5: [[ID:%.*]] = call i32 @llvm.amdgcn.workgroup.id.y()
// COV5: [[FULL:%.*]] = icmp ult i32 [[ID]], [[BC]]
// COV5: [[OFF:%.*]] = select i1 [[FULL]], i{{32|64}} 14, i{{32|64}} 20
// COV5: [[SGEP:%.*]] = getelementptr{{.*}} i8, ptr addrspace(4) [[IP]], i{{32|64}} [[OFF]]
// COV5: [[SZ:%.*]] = load i16, ptr addrspace(4) [[SGEP]], align 2
// COV5: zext i16 [[SZ]] to i32

// NONE-LABEL: @_Z4testv
// NONE: [[ABI:%.*]] = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4
// NONE: [[IS5:%.*]] = icmp sge i32 [[ABI]], 500
// NONE: call{{.*}} ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// NONE: [[V5:%.*]] = zext i16 %{{.*}} to i32
// NONE: call{{.*}} ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// NONE: [[V4:%.*]] = zext i16 %{{.*}} to i32
// NONE: select i1 [[IS5]], i32 [[V5]], i32 [[V4]]
__device__ unsigned test() { return __builtin_amdgcn_workgroup_size_y(); }
