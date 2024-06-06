// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

using pi1_t = int __attribute__((address_space(1))) *;
using pi2_t = int __attribute__((address_space(2))) *;

using ri1_t = int __attribute__((address_space(1))) &;
using ri2_t = int __attribute__((address_space(2))) &;

// CIR: cir.func @{{.*test_ptr.*}}
// LLVM: define void @{{.*test_ptr.*}}
void test_ptr() {
  pi1_t ptr1;
  pi2_t ptr2 = (pi2_t)ptr1;
  // CIR:      %[[#PTR1:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, addrspace(1)>>, !cir.ptr<!s32i, addrspace(1)>
  // CIR-NEXT: %[[#CAST:]] = cir.cast(address_space, %[[#PTR1]] : !cir.ptr<!s32i, addrspace(1)>), !cir.ptr<!s32i, addrspace(2)>
  // CIR-NEXT: cir.store %[[#CAST]], %{{[0-9]+}} : !cir.ptr<!s32i, addrspace(2)>, !cir.ptr<!cir.ptr<!s32i, addrspace(2)>>

  // LLVM:      %[[#PTR1:]] = load ptr addrspace(1), ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#PTR1]] to ptr addrspace(2)
  // LLVM-NEXT: store ptr addrspace(2) %[[#CAST]], ptr %{{[0-9]+}}, align 8
}

// CIR: cir.func @{{.*test_ref.*}}
// LLVM: define void @{{.*test_ref.*}}
void test_ref() {
  pi1_t ptr;
  ri1_t ref1 = *ptr;
  ri2_t ref2 = (ri2_t)ref1;
  // CIR:      %[[#DEREF:]] = cir.load deref %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, addrspace(1)>>, !cir.ptr<!s32i, addrspace(1)>
  // CIR-NEXT: cir.store %[[#DEREF]], %[[#ALLOCAREF1:]] : !cir.ptr<!s32i, addrspace(1)>, !cir.ptr<!cir.ptr<!s32i, addrspace(1)>>
  // CIR-NEXT: %[[#REF1:]] = cir.load %[[#ALLOCAREF1]] : !cir.ptr<!cir.ptr<!s32i, addrspace(1)>>, !cir.ptr<!s32i, addrspace(1)>
  // CIR-NEXT: %[[#CAST:]] = cir.cast(address_space, %[[#REF1]] : !cir.ptr<!s32i, addrspace(1)>), !cir.ptr<!s32i, addrspace(2)>
  // CIR-NEXT: cir.store %[[#CAST]], %{{[0-9]+}} : !cir.ptr<!s32i, addrspace(2)>, !cir.ptr<!cir.ptr<!s32i, addrspace(2)>>

  // LLVM:      %[[#DEREF:]] = load ptr addrspace(1), ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: store ptr addrspace(1) %[[#DEREF]], ptr %[[#ALLOCAREF1:]], align 8
  // LLVM-NEXT: %[[#REF1:]] = load ptr addrspace(1), ptr %[[#ALLOCAREF1]], align 8
  // LLVM-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#REF1]] to ptr addrspace(2)
  // LLVM-NEXT: store ptr addrspace(2) %[[#CAST]], ptr %{{[0-9]+}}, align 8
}

// CIR: cir.func @{{.*test_nullptr.*}}
// LLVM: define void @{{.*test_nullptr.*}}
void test_nullptr() {
  constexpr pi1_t null1 = nullptr;
  pi2_t ptr = (pi2_t)null1;
  // CIR:      %[[#NULL1:]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, addrspace(1)>
  // CIR-NEXT: cir.store %[[#NULL1]], %{{[0-9]+}} : !cir.ptr<!s32i, addrspace(1)>, !cir.ptr<!cir.ptr<!s32i, addrspace(1)>>
  // CIR-NEXT: %[[#NULL2:]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, addrspace(2)>
  // CIR-NEXT: cir.store %[[#NULL2]], %{{[0-9]+}} : !cir.ptr<!s32i, addrspace(2)>, !cir.ptr<!cir.ptr<!s32i, addrspace(2)>>

  // LLVM:      store ptr addrspace(1) null, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: store ptr addrspace(2) null, ptr %{{[0-9]+}}, align 8
}
