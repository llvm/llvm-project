// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

bool cmp_eq(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs == rhs;
}
  
// CIR-BEFORE: cir.func {{.*}} @_Z6cmp_eqM3FooFviES1_
// CIR-BEFORE:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[CMP:.*]] = cir.cmp(eq, %[[LHS]], %[[RHS]]) : !cir.method<!cir.func<(!s32i)> in !rec_Foo>, !cir.bool
// CIR-BEFORE:   cir.store %[[CMP]], %{{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// CIR-AFTER: @_Z6cmp_eqM3FooFviES1_
// CIR-AFTER:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[NULL:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER:   %[[LHS_PTR:.*]] = cir.extract_member %[[LHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[RHS_PTR:.*]] = cir.extract_member %[[RHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[PTR_CMP:.*]] = cir.cmp(eq, %[[LHS_PTR]], %[[RHS_PTR]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[PTR_NULL:.*]] = cir.cmp(eq, %[[LHS_PTR]], %[[NULL]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[LHS_ADJ:.*]] = cir.extract_member %[[LHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[RHS_ADJ:.*]] = cir.extract_member %[[RHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[ADJ_CMP:.*]] = cir.cmp(eq, %[[LHS_ADJ]], %[[RHS_ADJ]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[TMP:.*]] = cir.binop(or, %[[PTR_NULL]], %[[ADJ_CMP]]) : !cir.bool
// CIR-AFTER:   %[[RESULT:.*]] = cir.binop(and, %[[PTR_CMP]], %[[TMP]]) : !cir.bool

// LLVM: define {{.*}} i1 @_Z6cmp_eqM3FooFviES1_
// LLVM:   %[[LHS:.*]] = load { i64, i64 }, ptr %{{.+}}
// LLVM:   %[[RHS:.*]] = load { i64, i64 }, ptr %{{.+}}
// LLVM:   %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// LLVM:   %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// LLVM:   %[[PTR_CMP:.*]] = icmp eq i64 %[[LHS_PTR]], %[[RHS_PTR]]
// LLVM:   %[[PTR_NULL:.*]] = icmp eq i64 %[[LHS_PTR]], 0
// LLVM:   %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// LLVM:   %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// LLVM:   %[[ADJ_CMP:.*]] = icmp eq i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM:   %[[TMP:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// LLVM:   %[[RESULT:.*]] = and i1 %[[PTR_CMP]], %[[TMP]]

// OGCG: define {{.*}} i1 @_Z6cmp_eqM3FooFviES1_
// OGCG:   %[[LHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:   %[[RHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:   %[[LHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   %[[RHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   %[[LHS:.*]] = load { i64, i64 }, ptr %[[LHS_ADDR]]
// OGCG:   %[[RHS:.*]] = load { i64, i64 }, ptr %[[RHS_ADDR]]
// OGCG:   %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// OGCG:   %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// OGCG:   %[[PTR_CMP:.*]] = icmp eq i64 %[[LHS_PTR]], %[[RHS_PTR]]
// OGCG:   %[[PTR_NULL:.*]] = icmp eq i64 %[[LHS_PTR]], 0
// OGCG:   %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// OGCG:   %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// OGCG:   %[[ADJ_CMP:.*]] = icmp eq i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG:   %[[TMP:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// OGCG:   %[[RESULT:.*]] = and i1 %[[PTR_CMP]], %[[TMP]]

bool cmp_ne(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs != rhs;
}
  
// CIR-BEFORE: cir.func {{.*}} @_Z6cmp_neM3FooFviES1_
// CIR-BEFORE:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[CMP:.*]] = cir.cmp(ne, %[[LHS]], %[[RHS]]) : !cir.method<!cir.func<(!s32i)> in !rec_Foo>, !cir.bool
// CIR-BEFORE:   cir.store %[[CMP]], %{{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// CIR-AFTER: cir.func {{.*}} @_Z6cmp_neM3FooFviES1_
// CIR-AFTER:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[NULL:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER:   %[[LHS_PTR:.*]] = cir.extract_member %[[LHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[RHS_PTR:.*]] = cir.extract_member %[[RHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[PTR_CMP:.*]] = cir.cmp(ne, %[[LHS_PTR]], %[[RHS_PTR]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[PTR_NULL:.*]] = cir.cmp(ne, %[[LHS_PTR]], %[[NULL]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[LHS_ADJ:.*]] = cir.extract_member %[[LHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[RHS_ADJ:.*]] = cir.extract_member %[[RHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[ADJ_CMP:.*]] = cir.cmp(ne, %[[LHS_ADJ]], %[[RHS_ADJ]]) : !s64i, !cir.bool
// CIR-AFTER:   %[[TMP:.*]] = cir.binop(and, %[[PTR_NULL]], %[[ADJ_CMP]]) : !cir.bool
// CIR-AFTER:   %[[RESULT:.*]] = cir.binop(or, %[[PTR_CMP]], %[[TMP]]) : !cir.bool

// LLVM: define {{.*}} i1 @_Z6cmp_neM3FooFviES1_
// LLVM:   %[[LHS:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   %[[RHS:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// LLVM:   %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// LLVM:   %[[PTR_CMP:.*]] = icmp ne i64 %[[LHS_PTR]], %[[RHS_PTR]]
// LLVM:   %[[PTR_NULL:.*]] = icmp ne i64 %[[LHS_PTR]], 0
// LLVM:   %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// LLVM:   %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// LLVM:   %[[ADJ_CMP:.*]] = icmp ne i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM:   %[[TMP:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// LLVM:   %[[RESULT:.*]] = or i1 %[[PTR_CMP]], %[[TMP]]

// OGCG: define {{.*}} i1 @_Z6cmp_neM3FooFviES1_
// OGCG:   %[[LHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:   %[[RHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:   %[[LHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   %[[RHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   %[[LHS:.*]] = load { i64, i64 }, ptr %[[LHS_ADDR]]
// OGCG:   %[[RHS:.*]] = load { i64, i64 }, ptr %[[RHS_ADDR]]
// OGCG:   %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// OGCG:   %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// OGCG:   %[[PTR_CMP:.*]] = icmp ne i64 %[[LHS_PTR]], %[[RHS_PTR]]
// OGCG:   %[[PTR_NULL:.*]] = icmp ne i64 %[[LHS_PTR]], 0
// OGCG:   %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// OGCG:   %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// OGCG:   %[[ADJ_CMP:.*]] = icmp ne i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG:   %[[TMP:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// OGCG:   %[[RESULT:.*]] = or i1 %[[PTR_CMP]], %[[TMP]]
