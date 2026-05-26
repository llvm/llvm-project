// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// Padding-only union: CIR keeps only a tail pad member, so
// getLargestMember() is null and getABIAlignment must not call
// getTypeABIAlignment on null.
union Empty {};
// CIR: !rec_Empty = !cir.record<union "Empty" padded {!u8i}>
// LLVM-DAG: %union.Empty = type { i8 }
// OGCG-DAG: %union.Empty = type { i8 }

// Aligned empty union (should have aligned integer member in CIR)
union alignas(16) EmptyAligned {};
// CIR: !rec_EmptyAligned = !cir.record<union "EmptyAligned" padded {!cir.array<!u8i x 16>}>
// LLVM-DAG: %union.EmptyAligned = type { [16 x i8] }
// OGCG-DAG: %union.EmptyAligned = type { [16 x i8] }

// Struct holding a padding-only union member: layout queries !rec_Empty
// alignment (null largest), not OuterWithEmpty's int x.
union OuterWithEmpty {
  Empty e;
  int x;
};
struct WrapEmpty {
  OuterWithEmpty o;
  int s;
};
WrapEmpty w;
// CIR:      !rec_OuterWithEmpty = !cir.record<union "OuterWithEmpty" {!rec_Empty, !s32i}>
// CIR:      !rec_WrapEmpty = !cir.record<struct "WrapEmpty" {!rec_OuterWithEmpty, !s32i}>
// CIR:      cir.global external @w = #cir.zero : !rec_WrapEmpty {alignment = 4 : i64}
// LLVM-DAG: %struct.WrapEmpty = type { %union.OuterWithEmpty, i32 }
// LLVM-DAG: %union.OuterWithEmpty = type { i32 }
// LLVM:     @w = global %struct.WrapEmpty zeroinitializer, align 4
// OGCG-DAG: %struct.WrapEmpty = type { %union.OuterWithEmpty, i32 }
// OGCG-DAG: %union.OuterWithEmpty = type { i32 }
// OGCG:     @w = global %struct.WrapEmpty zeroinitializer, align 4

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
