// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// Empty union (should be padded to size 1)
union Empty {};
// CIR: ![[INT128:.*]] = !cir.int<u, 128>
// CIR: !rec_Empty = !cir.record<union "Empty" {!u8i}>
// LLVM: %union.Empty = type { i8 }

// Aligned empty union (should have aligned integer member in CIR)
union alignas(16) EmptyAligned {};
// CIR: !rec_EmptyAligned = !cir.record<union "EmptyAligned" {![[INT128]]}>
// LLVM: %union.EmptyAligned = type { i128 }

void useEmpty() {
  Empty e;
}
// CIR: cir.func {{.*}}@_Z8useEmptyv()
// CIR:   cir.alloca !rec_Empty, !cir.ptr<!rec_Empty>, ["e"] {alignment = 1 : i64}
// LLVM: define {{.*}} void @_Z8useEmptyv()
// LLVM:   alloca %union.Empty, i64 1, align 1

void useEmptyAligned() {
  EmptyAligned e;
}
// CIR: cir.func {{.*}}@_Z15useEmptyAlignedv()
// CIR:   cir.alloca !rec_EmptyAligned, !cir.ptr<!rec_EmptyAligned>, ["e"] {alignment = 16 : i64}
// LLVM: define {{.*}} void @_Z15useEmptyAlignedv()
// LLVM:   alloca %union.EmptyAligned, i64 1, align 16
