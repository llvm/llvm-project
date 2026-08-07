// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// Empty union (no padding, size 0 in CIR)
union Empty {};
// CIR: !rec_Empty = !cir.union<"Empty" {}>
// LLVM: %union.Empty = type {}
// OGCG: %union.Empty = type {}

// Aligned empty union (size 0, alignment 16)
union EmptyAligned {} __attribute__((aligned(16)));
// CIR: !rec_EmptyAligned = !cir.union<"EmptyAligned" {}>
// LLVM: %union.EmptyAligned = type {}
// OGCG: %union.EmptyAligned = type {}

void useEmpty() {
  union Empty e;
}
// CIR: cir.func {{.*}}@useEmpty()
// CIR:   cir.alloca "e" align(1) : !cir.ptr<!rec_Empty>
// LLVM: define {{.*}} void @useEmpty()
// LLVM:   alloca %union.Empty, i64 1, align 1
// OGCG: define {{.*}} void @useEmpty()
// OGCG:   alloca %union.Empty, align 1

void useEmptyAligned() {
  union EmptyAligned e;
}
// CIR: cir.func {{.*}}@useEmptyAligned()
// CIR:   cir.alloca "e" align(16) : !cir.ptr<!rec_EmptyAligned>
// LLVM: define {{.*}} void @useEmptyAligned()
// LLVM:   alloca %union.EmptyAligned, i64 1, align 16
// OGCG: define {{.*}} void @useEmptyAligned()
// OGCG:   alloca %union.EmptyAligned, align 16
