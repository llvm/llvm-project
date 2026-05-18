// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: FileCheck --check-prefix=CIR-NUA --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

struct Base {
  int x;
};

struct Middle : Base {
  char c;
  // sizeof(Middle) = 8 (4 for x, 1 for c, 3 tail padding)
  // data size = 5
};

struct Outer {
  [[no_unique_address]] Middle m;
  char extra;
  Outer(const Middle &m, char e) : m(m), extra(e) {}
};

// The record layout should use the base subobject type for the
// [[no_unique_address]] field, allowing 'extra' to overlap with
// Middle's tail padding.

// CIR: !rec_Middle2Ebase = !cir.record<struct "Middle.base" packed {!rec_Base, !s8i}>
// CIR: !rec_Outer = !cir.record<struct "Outer" padded {!rec_Middle2Ebase, !s8i,

// CIR-LABEL: cir.func {{.*}} @_ZN5OuterC2ERK6Middlec(
// CIR:         %[[THIS:.*]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!rec_Outer>>, !cir.ptr<!rec_Outer>
// CIR:         %[[M_BASE:.*]] = cir.get_member %[[THIS]][0] {name = "m"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_Middle2Ebase>
// CIR-NEXT:    %[[M_COMPLETE:.*]] = cir.cast bitcast %[[M_BASE]] : !cir.ptr<!rec_Middle2Ebase> -> !cir.ptr<!rec_Middle>
// CIR:         cir.copy %{{.+}} to %[[M_COMPLETE]] skip_tail_padding : !cir.ptr<!rec_Middle>
// CIR:         %[[EXTRA:.*]] = cir.get_member %[[THIS]][1] {name = "extra"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!s8i>

// Globals for the union/final NUA cases below (placed before LLVM-LABEL so
// these DAG checks anchor to the top of the .ll file rather than to the
// function body).
// LLVM-DAG: %struct.OuterUnion = type { %union.UnionForNUA, i32 }
// LLVM-DAG: %union.UnionForNUA = type { i64 }
// LLVM-DAG: %struct.OuterFinal = type { %struct.FinalForNUA, i8 }
// LLVM-DAG: %struct.FinalForNUA = type { i32, i8 }
// LLVM-DAG: @ou = {{(dso_local )?}}global %struct.OuterUnion zeroinitializer, align 8
// LLVM-DAG: @of = {{(dso_local )?}}global %struct.OuterFinal zeroinitializer, align 4
// OGCG-DAG: %struct.OuterUnion = type { %union.UnionForNUA, i32 }
// OGCG-DAG: %union.UnionForNUA = type { i64 }
// OGCG-DAG: %struct.OuterFinal = type { %struct.FinalForNUA, i8 }
// OGCG-DAG: %struct.FinalForNUA = type { i32, i8 }
// OGCG-DAG: @ou = {{(dso_local )?}}global %struct.OuterUnion zeroinitializer, align 8
// OGCG-DAG: @of = {{(dso_local )?}}global %struct.OuterFinal zeroinitializer, align 4

// LLVM-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// LLVM:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr %[[GEP]], ptr %{{.+}}, i64 5, i1 false)

// OGCG-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// OGCG:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[GEP]], ptr {{.*}} %{{.+}}, i64 5, i1 false)

void test(const Middle &m) {
  Outer o(m, 'x');
}

// Regression test: a [[no_unique_address]] field whose type is a union (or a
// final class) used to crash CIRRecordLowering with an empty SmallVector::back()
// because computeRecordLayout left the base-subobject type unset for those
// kinds of records, and getStorageType(const CXXRecordDecl *) propagated the
// resulting null mlir::Type into the members vector. We now set baseTy = ty
// for all C++ records, so these layouts succeed.

union UnionForNUA {
  int i;
  long l;
};

struct OuterUnion {
  [[no_unique_address]] UnionForNUA u;
  int x;
};

OuterUnion ou;

struct FinalForNUA final {
  int a;
  char b;
};

struct OuterFinal {
  [[no_unique_address]] FinalForNUA f;
  char tail;
};

OuterFinal of;

// CIR-NUA-DAG: !rec_FinalForNUA = !cir.record<struct "FinalForNUA" {!s32i, !s8i}>
// CIR-NUA-DAG: !rec_UnionForNUA = !cir.record<union "UnionForNUA" {!s32i, !s64i}>
// CIR-NUA-DAG: !rec_OuterFinal = !cir.record<struct "OuterFinal" {!rec_FinalForNUA, !s8i}>
// CIR-NUA-DAG: !rec_OuterUnion = !cir.record<struct "OuterUnion" {!rec_UnionForNUA, !s32i}>
// CIR-NUA-DAG: cir.global external @ou = #cir.zero : !rec_OuterUnion
// CIR-NUA-DAG: cir.global external @of = #cir.zero : !rec_OuterFinal

