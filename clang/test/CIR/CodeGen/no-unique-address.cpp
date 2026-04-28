// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
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

// LLVM-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// LLVM:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr %[[GEP]], ptr %{{.+}}, i64 5, i1 false)

// OGCG-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// OGCG:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[GEP]], ptr {{.*}} %{{.+}}, i64 5, i1 false)

void test(const Middle &m) {
  Outer o(m, 'x');
}
