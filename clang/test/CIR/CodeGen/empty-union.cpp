// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// Empty union (should be padded to size 1)
union Empty {};
// CIR: !rec_Empty = !cir.record<union "Empty" padded {!u8i}>
// LLVM: %union.Empty = type { i8 }
// OGCG: %union.Empty = type { i8 }

// Aligned empty union (should have aligned integer member in CIR)
union alignas(16) EmptyAligned {};
// CIR: !rec_EmptyAligned = !cir.record<union "EmptyAligned" padded {!cir.array<!u8i x 16>}>
// LLVM: %union.EmptyAligned = type { [16 x i8] }
// OGCG: %union.EmptyAligned = type { [16 x i8] }

void useEmpty() {
  Empty e;
}
// CIR: cir.func {{.*}}@_Z8useEmptyv()
// CIR:   cir.alloca !rec_Empty, !cir.ptr<!rec_Empty>, ["e"] {alignment = 1 : i64}
// LLVM: define {{.*}} void @_Z8useEmptyv()
// LLVM:   alloca %union.Empty, i64 1, align 1
// OGCG: define {{.*}} void @_Z8useEmptyv()
// OGCG:   alloca %union.Empty, align 1

void useEmptyAligned() {
  EmptyAligned e;
}
// CIR: cir.func {{.*}}@_Z15useEmptyAlignedv()
// CIR:   cir.alloca !rec_EmptyAligned, !cir.ptr<!rec_EmptyAligned>, ["e"] {alignment = 16 : i64}
// LLVM: define {{.*}} void @_Z15useEmptyAlignedv()
// LLVM:   alloca %union.EmptyAligned, i64 1, align 16
// OGCG: define {{.*}} void @_Z15useEmptyAlignedv()
// OGCG:   alloca %union.EmptyAligned, align 16
