#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-CUDA
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-CUDA
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-HIP
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-HIP
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=OGCG,OGCG-CUDA
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=OGCG,OGCG-HIP

// The loop variable of a lowered array constructor or destructor is a stack
// local, so it lives in the alloca address space, aligned for the element
// pointer it holds. Classic codegen keeps it in a phi instead.

struct S {
  __device__ S();
  __device__ ~S();
  int x;
};

__device__ void local_array() { S arr[4]; }

// CIR-CUDA-LABEL: cir.func {{.*}} @_Z11local_arrayv()
// CIR-CUDA:   %[[CTOR_IDX:.*]] = cir.alloca "__array_idx" align(8) : !cir.ptr<!cir.ptr<!rec_S>>
// CIR-CUDA:   cir.do {
// CIR-CUDA:     %[[CUR:.*]] = cir.load %[[CTOR_IDX]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR-CUDA:     cir.call @_ZN1SC1Ev(%[[CUR]])
// CIR-CUDA:   %[[DTOR_IDX:.*]] = cir.alloca "__array_idx" align(8) : !cir.ptr<!cir.ptr<!rec_S>>
// CIR-CUDA:   cir.do {
// CIR-CUDA:     cir.load %[[DTOR_IDX]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR-CUDA:     cir.call @_ZN1SD1Ev(

// LLVM-CUDA-LABEL: define {{.*}} void @_Z11local_arrayv()
// LLVM-CUDA:   %[[DTOR_IDX:.*]] = alloca ptr, align 8
// LLVM-CUDA:   %[[CTOR_IDX:.*]] = alloca ptr, align 8
// LLVM-CUDA:   store ptr %{{.*}}, ptr %[[CTOR_IDX]], align 8
// LLVM-CUDA:   load ptr, ptr %[[CTOR_IDX]], align 8
// LLVM-CUDA:   call void @_ZN1SC1Ev(
// LLVM-CUDA:   store ptr %{{.*}}, ptr %[[DTOR_IDX]], align 8
// LLVM-CUDA:   load ptr, ptr %[[DTOR_IDX]], align 8
// LLVM-CUDA:   call void @_ZN1SD1Ev(

// CIR-HIP-LABEL: cir.func {{.*}} @_Z11local_arrayv()
// CIR-HIP:   %[[CTOR_IDX:.*]] = cir.alloca "__array_idx" align(8) : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>
// CIR-HIP:   cir.do {
// CIR-HIP:     %[[CUR:.*]] = cir.load %[[CTOR_IDX]] : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>, !cir.ptr<!rec_S>
// CIR-HIP:     cir.call @_ZN1SC1Ev(%[[CUR]])
// CIR-HIP:   %[[DTOR_IDX:.*]] = cir.alloca "__array_idx" align(8) : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>
// CIR-HIP:   cir.do {
// CIR-HIP:     cir.load %[[DTOR_IDX]] : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>, !cir.ptr<!rec_S>
// CIR-HIP:     cir.call @_ZN1SD1Ev(

// LLVM-HIP-LABEL: define {{.*}} void @_Z11local_arrayv()
// LLVM-HIP:   %[[DTOR_IDX:.*]] = alloca ptr, align 8, addrspace(5)
// LLVM-HIP:   %[[CTOR_IDX:.*]] = alloca ptr, align 8, addrspace(5)
// LLVM-HIP:   store ptr %{{.*}}, ptr addrspace(5) %[[CTOR_IDX]], align 8
// LLVM-HIP:   load ptr, ptr addrspace(5) %[[CTOR_IDX]], align 8
// LLVM-HIP:   call void @_ZN1SC1Ev(
// LLVM-HIP:   store ptr %{{.*}}, ptr addrspace(5) %[[DTOR_IDX]], align 8
// LLVM-HIP:   load ptr, ptr addrspace(5) %[[DTOR_IDX]], align 8
// LLVM-HIP:   call void @_ZN1SD1Ev(

// OGCG-LABEL: define {{.*}} void @_Z11local_arrayv()
// OGCG-CUDA:    alloca [4 x %struct.S], align 4{{$}}
// OGCG-HIP:     alloca [4 x %struct.S], align 4, addrspace(5)
// OGCG:       arrayctor.loop:
// OGCG:         %[[CUR:.*]] = phi ptr
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[CUR]])
// OGCG:       arraydestroy.body:
// OGCG:         %[[PAST:.*]] = phi ptr
// OGCG:         %[[ELT:.*]] = getelementptr inbounds %struct.S, ptr %[[PAST]], i64 -1
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[ELT]])
