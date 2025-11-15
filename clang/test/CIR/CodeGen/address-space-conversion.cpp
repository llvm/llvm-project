// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

using pi1_t = int __attribute__((address_space(1))) *;
using pi2_t = int __attribute__((address_space(2))) *;

using ri1_t = int __attribute__((address_space(1))) &;
using ri2_t = int __attribute__((address_space(2))) &;

// CIR: cir.func dso_local @{{.*test_ptr.*}}
// LLVM: define dso_local void @{{.*test_ptr.*}}
// OGCG: define dso_local void @{{.*test_ptr.*}}
void test_ptr() {
  pi1_t ptr1;
  pi2_t ptr2 = (pi2_t)ptr1;
  // CIR:      %[[#PTR1:]] = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>, !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: %[[#CAST:]] = cir.cast address_space %[[#PTR1]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: cir.store{{.*}} %[[#CAST]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(2)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(2)>>

  // LLVM:      %[[#PTR1:]] = load ptr addrspace(1), ptr %{{.*}}
  // LLVM-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#PTR1]] to ptr addrspace(2)
  // LLVM-NEXT: store ptr addrspace(2) %[[#CAST]], ptr %{{.*}}

  // OGCG:      %{{.*}} = load ptr addrspace(1), ptr %{{.*}}
  // OGCG-NEXT: %{{.*}} = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(2)
  // OGCG-NEXT: store ptr addrspace(2)  %{{.*}}, ptr %{{.*}}
}

// CIR: cir.func dso_local @{{.*test_ref.*}}
// LLVM: define dso_local void @{{.*test_ref.*}}
// OGCG: define dso_local void @{{.*test_ref.*}}
void test_ref() {
  pi1_t ptr;
  ri1_t ref1 = *ptr;
  ri2_t ref2 = (ri2_t)ref1;
  // CIR:      %[[#DEREF:]] = cir.load deref{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>, !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: cir.store{{.*}} %[[#DEREF]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(1)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
  // CIR-NEXT: %[[#REF1:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>, !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: %[[#CAST:]] = cir.cast address_space %[[#REF1]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: cir.store{{.*}} %[[#CAST]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(2)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(2)>>

  // LLVM:      %[[#DEREF:]] = load ptr addrspace(1), ptr %{{.*}}
  // LLVM-NEXT: store ptr addrspace(1) %[[#DEREF]], ptr %{{.*}}
  // LLVM-NEXT: %[[#REF1:]] = load ptr addrspace(1), ptr %{{.*}}
  // LLVM-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#REF1]] to ptr addrspace(2)
  // LLVM-NEXT: store ptr addrspace(2) %[[#CAST]], ptr %{{.*}}

  // OGCG:      %{{.*}} = load ptr addrspace(1), ptr %{{.*}}
  // OGCG-NEXT: store ptr addrspace(1) %{{.*}}, ptr %{{.*}}
  // OGCG-NEXT: %{{.*}} = load ptr addrspace(1), ptr %{{.*}}
  // OGCG-NEXT: %{{.*}} = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(2)
  // OGCG-NEXT: store ptr addrspace(2) %{{.*}}, ptr %{{.*}}
}

// CIR: cir.func dso_local @{{.*test_nullptr.*}}
// LLVM: define dso_local void @{{.*test_nullptr.*}}
// OGCG: define dso_local void @{{.*test_nullptr.*}}
void test_nullptr() {
  constexpr pi1_t null1 = nullptr;
  pi2_t ptr = (pi2_t)null1;
  // CIR:      %[[#NULL1:]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: cir.store{{.*}} %[[#NULL1]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(1)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
  // CIR-NEXT: %[[#NULL2:]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: cir.store{{.*}} %[[#NULL2]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(2)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(2)>>

  // LLVM:      store ptr addrspace(1) null, ptr %{{.*}}
  // LLVM-NEXT: store ptr addrspace(2) null, ptr %{{.*}}

  // OGCG:      store ptr addrspace(1) null, ptr %{{.*}}
  // OGCG-NEXT: store ptr addrspace(2) null, ptr %{{.*}}
}

// CIR: cir.func dso_local @{{.*test_side_effect.*}}
// LLVM: define dso_local void @{{.*test_side_effect.*}}
// OGCG: define dso_local void @{{.*test_side_effect.*}}
void test_side_effect(pi1_t b) {
  pi2_t p = (pi2_t)(*b++, (int*)0);
  // CIR:      %[[#DEREF:]] = cir.load deref{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>, !cir.ptr<!s32i, target_address_space(1)>
  // CIR:      %[[#STRIDE:]] = cir.ptr_stride %[[#DEREF]], %{{[0-9]+}} : (!cir.ptr<!s32i, target_address_space(1)>, !s32i) -> !cir.ptr<!s32i, target_address_space(1)>
  // CIR:      %[[#NULL:]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: cir.store{{.*}} %[[#NULL]], %{{[0-9]+}} : !cir.ptr<!s32i, target_address_space(2)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(2)>>

  // LLVM:      %{{[0-9]+}} = getelementptr {{.*}}i32, ptr addrspace(1) %{{[0-9]+}}, i{{32|64}} 1
  // LLVM:      store ptr addrspace(2) null, ptr %{{.*}}

  // OGCG:      %{{.*}} = getelementptr{{.*}} i32, ptr addrspace(1) %{{.*}}, i32 1
  // OGCG:      store ptr addrspace(2) null, ptr %{{.*}}
}
