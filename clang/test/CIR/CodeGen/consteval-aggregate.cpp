// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ogcg.ll %s -check-prefix=OGCG

struct Agg { int a; long b; };
consteval Agg retAgg() { return {13, 17}; }

long test_retAgg() {
  long b = retAgg().b;
  return b;
}

// CIR-LABEL: @_Z11test_retAggv
// CIR:   cir.get_global @__const._Z11test_retAggv.ref.tmp0 : !cir.ptr<!rec_Agg>
// CIR:   cir.copy

// TODO(CIR): CIR materializes consteval aggregates as global constants and
// uses memcpy, while OGCG inlines the stores directly. This should be unified
// to match OGCG's behavior for small aggregates.
// LLVM-LABEL: @_Z11test_retAggv
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr @__const._Z11test_retAggv.ref.tmp0, i64 16, i1 false)

// OGCG-LABEL: @_Z11test_retAggv
// OGCG:   store i32 13, ptr %{{.*}}, align 8
// OGCG:   store i64 17, ptr %{{.*}}, align 8

int test_retAgg_first() {
  int a = retAgg().a;
  return a;
}

// CIR-LABEL: @_Z17test_retAgg_firstv
// CIR:   cir.get_global @__const._Z17test_retAgg_firstv.ref.tmp0 : !cir.ptr<!rec_Agg>
// CIR:   cir.copy

// LLVM-LABEL: @_Z17test_retAgg_firstv
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr @__const._Z17test_retAgg_firstv.ref.tmp0, i64 16, i1 false)

// OGCG-LABEL: @_Z17test_retAgg_firstv
// OGCG:   store i32 13, ptr %{{.*}}, align 8
// OGCG:   store i64 17, ptr %{{.*}}, align 8
