// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Regression test: a [[no_unique_address]] empty member/base initialized by
// a consteval constructor is folded by Sema into a ConstantExpr. CIRGen's
// AggExprEmitter::VisitConstantExpr used to unconditionally store the
// folded (zero-size) constant at the member's address. Because a truly
// empty [[no_unique_address]] subobject overlaps the storage of an earlier
// field, that store clobbered the field's just-initialized value.
// VisitConstantExpr must honor the destination's overlap flag and skip the
// store entirely when the type's data size is zero.
//
// The CHECK blocks below are ordered to match the actual emission order
// (base-object '*C2' constructors are grouped together after all the
// '*C1' complete-object constructors and free functions), not the source
// order of the structs.

struct Empty {
  consteval Empty(int) {}
};

// Member case: 'e' is empty and [[no_unique_address]], so it is laid out at
// offset 0, overlapping 'v'. Initializing 'e' must not touch 'v's bytes.
struct S {
  unsigned long v;
  [[no_unique_address]] Empty e;
  S(unsigned long x) : v(x), e(0) {}
};

unsigned long test_member() {
  S s(0x1234);
  return s.v;
}

// Base case: EmptyBase is empty, so empty-base optimization places it at the
// same address as 'v2'.
struct EmptyBase {
  consteval EmptyBase(int) {}
};

struct T : EmptyBase {
  unsigned long v2;
  T(unsigned long x) : EmptyBase(0), v2(x) {}
};

unsigned long test_base() {
  T t(0x5678);
  return t.v2;
}

// Aggregate-initialization case: the init-list path already requests
// MayOverlap for TEK_Aggregate fields, so this exercises the same fix in
// VisitConstantExpr via a different caller.
struct U {
  unsigned long v3;
  [[no_unique_address]] Empty e;
};

// CIR-LABEL: cir.func {{.*}} @_Z19test_aggregate_initv(
// CIR:         %[[V3:.*]] = cir.get_member %{{.+}}[0] {name = "v3"}
// CIR-NEXT:    %[[C:.*]] = cir.const #cir.int<39612>
// CIR-NEXT:    cir.store align(8) %[[C]], %[[V3]] : !u64i, !cir.ptr<!u64i>
// CIR-NOT:      #cir.zero : !rec_Empty

// LLVM-LABEL: define {{.*}} i64 @_Z19test_aggregate_initv(
// LLVM:         %[[V3:.*]] = getelementptr inbounds nuw %struct.U, ptr %{{.+}}, i32 0, i32 0
// LLVM-NEXT:    store i64 39612, ptr %[[V3]], align 8

// OGCG-LABEL: define {{.*}} i64 @_Z19test_aggregate_initv(
// OGCG:         %[[V3:.*]] = getelementptr inbounds nuw %struct.U, ptr %{{.+}}, i32 0, i32 0
// OGCG-NEXT:    store i64 39612, ptr %[[V3]], align 8

unsigned long test_aggregate_init() {
  U u{0x9abc, 0};
  return u.v3;
}

// Default-member-initializer case.
struct V {
  unsigned long v4;
  [[no_unique_address]] Empty e = Empty(0);
  V(unsigned long x) : v4(x) {}
};

unsigned long test_default_member_init() {
  V v(0xdef0);
  return v.v4;
}

// CIR-LABEL: cir.func {{.*}} @_ZN1SC2Em(
// CIR:         %[[V:.*]] = cir.get_member %{{.+}}[0] {name = "v"}
// CIR:         %[[X:.*]] = cir.load align(8) %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.store align(8) %[[X]], %[[V]] : !u64i, !cir.ptr<!u64i>
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} void @_ZN1SC2Em(
// LLVM:         %[[V:.*]] = getelementptr inbounds nuw %struct.S, ptr %{{.+}}, i32 0, i32 0
// LLVM:         %[[X:.*]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    store i64 %[[X]], ptr %[[V]], align 8
// LLVM-NEXT:    ret void

// OGCG-LABEL: define {{.*}} void @_ZN1SC2Em(
// OGCG:         %[[V:.*]] = getelementptr inbounds nuw %struct.S, ptr %{{.+}}, i32 0, i32 0
// OGCG:         %[[X:.*]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    store i64 %[[X]], ptr %[[V]], align 8
// OGCG-NEXT:    ret void

// CIR-LABEL: cir.func {{.*}} @_ZN1TC2Em(
// CIR:         cir.base_class_addr
// CIR:         %[[V2:.*]] = cir.get_member %{{.+}}[0] {name = "v2"}
// CIR:         %[[X2:.*]] = cir.load align(8) %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.store align(8) %[[X2]], %[[V2]] : !u64i, !cir.ptr<!u64i>
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} void @_ZN1TC2Em(
// LLVM:         %[[V2:.*]] = getelementptr inbounds nuw %struct.T, ptr %{{.+}}, i32 0, i32 0
// LLVM:         %[[X2:.*]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    store i64 %[[X2]], ptr %[[V2]], align 8
// LLVM-NEXT:    ret void

// OGCG-LABEL: define {{.*}} void @_ZN1TC2Em(
// OGCG:         %[[V2:.*]] = getelementptr inbounds nuw %struct.T, ptr %{{.+}}, i32 0, i32 0
// OGCG:         %[[X2:.*]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    store i64 %[[X2]], ptr %[[V2]], align 8
// OGCG-NEXT:    ret void

// CIR-LABEL: cir.func {{.*}} @_ZN1VC2Em(
// CIR:         %[[V4:.*]] = cir.get_member %{{.+}}[0] {name = "v4"}
// CIR:         %[[X4:.*]] = cir.load align(8) %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.store align(8) %[[X4]], %[[V4]] : !u64i, !cir.ptr<!u64i>
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} void @_ZN1VC2Em(
// LLVM:         %[[V4:.*]] = getelementptr inbounds nuw %struct.V, ptr %{{.+}}, i32 0, i32 0
// LLVM:         %[[X4:.*]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    store i64 %[[X4]], ptr %[[V4]], align 8
// LLVM-NEXT:    ret void

// OGCG-LABEL: define {{.*}} void @_ZN1VC2Em(
// OGCG:         %[[V4:.*]] = getelementptr inbounds nuw %struct.V, ptr %{{.+}}, i32 0, i32 0
// OGCG:         %[[X4:.*]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    store i64 %[[X4]], ptr %[[V4]], align 8
// OGCG-NEXT:    ret void
