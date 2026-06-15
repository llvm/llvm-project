// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-X86 --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefixes=LLVM,LLVM-X86 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG,OGCG-X86 --input-file=%t.ll %s

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t-arm.cir 2> %t-arm-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-arm-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-ARM --input-file=%t-arm.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-arm-cir.ll
// RUN: FileCheck --input-file=%t-arm-cir.ll --check-prefixes=LLVM,LLVM-ARM %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t-arm.ll
// RUN: FileCheck --check-prefixes=OGCG,OGCG-ARM --input-file=%t-arm.ll %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

bool cmp_eq(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs == rhs;
}

// CIR-BEFORE: cir.func {{.*}} @_Z6cmp_eqM3FooFviES1_
// CIR-BEFORE:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[CMP:.*]] = cir.cmp eq %[[LHS]], %[[RHS]] : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[CMP]], %{{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// CIR-AFTER:     @_Z6cmp_eqM3FooFviES1_
// CIR-AFTER:       %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:       %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:       %[[NULL:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER:       %[[LHS_PTR:.*]] = cir.extract_member %[[LHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[RHS_PTR:.*]] = cir.extract_member %[[RHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[PTR_CMP:.*]] = cir.cmp eq %[[LHS_PTR]], %[[RHS_PTR]] : !s64i
// CIR-AFTER:       %[[PTR_NULL:.*]] = cir.cmp eq %[[LHS_PTR]], %[[NULL]] : !s64i
// CIR-AFTER:       %[[LHS_ADJ:.*]] = cir.extract_member %[[LHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[RHS_ADJ:.*]] = cir.extract_member %[[RHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[ADJ_CMP:.*]] = cir.cmp eq %[[LHS_ADJ]], %[[RHS_ADJ]] : !s64i
// CIR-AFTER-X86:   %[[TMP:.*]] = cir.or %[[PTR_NULL]], %[[ADJ_CMP]] : !cir.bool
// CIR-AFTER-ARM:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-AFTER-ARM:   %[[OR_ADJ:.*]] = cir.or %[[LHS_ADJ]], %[[RHS_ADJ]] : !s64i
// CIR-AFTER-ARM:   %[[AND_ADJ:.*]] = cir.and %[[OR_ADJ]], %[[ONE]] : !s64i
// CIR-AFTER-ARM:   %[[ADJ_CMP2:.*]] = cir.cmp eq %[[AND_ADJ]], %[[NULL]] : !s64i
// CIR-AFTER-ARM:   %[[AND_PTR_NULL:.*]] = cir.and %[[PTR_NULL]], %[[ADJ_CMP2]] : !cir.bool
// CIR-AFTER-ARM:   %[[TMP:.*]] = cir.or %[[AND_PTR_NULL]], %[[ADJ_CMP]] : !cir.bool
// CIR-AFTER:       %[[RESULT:.*]] = cir.and %[[PTR_CMP]], %[[TMP]] : !cir.bool

// LLVM:     define {{.*}} i1 @_Z6cmp_eqM3FooFviES1_
// LLVM:       %[[LHS:.*]] = load { i64, i64 }, ptr %{{.+}}
// LLVM:       %[[RHS:.*]] = load { i64, i64 }, ptr %{{.+}}
// LLVM:       %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// LLVM:       %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// LLVM:       %[[PTR_CMP:.*]] = icmp eq i64 %[[LHS_PTR]], %[[RHS_PTR]]
// LLVM:       %[[PTR_NULL:.*]] = icmp eq i64 %[[LHS_PTR]], 0
// LLVM:       %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// LLVM:       %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// LLVM:       %[[ADJ_CMP:.*]] = icmp eq i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM-X86:   %[[TMP:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// LLVM-ARM:   %[[OR_ADJ:.*]] = or i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM-ARM:   %[[AND_ADJ:.*]] = and i64 %[[OR_ADJ]], 1
// LLVM-ARM:   %[[ADJ_CMP2:.*]] = icmp eq i64 %[[AND_ADJ]], 0
// LLVM-ARM:   %[[AND_PTR_NULL:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP2]]
// LLVM-ARM:   %[[TMP:.*]] = or i1 %[[AND_PTR_NULL]], %[[ADJ_CMP]]
// LLVM:       %[[RESULT:.*]] = and i1 %[[PTR_CMP]], %[[TMP]]

// OGCG:     define {{.*}} i1 @_Z6cmp_eqM3FooFviES1_
// OGCG:       %[[LHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:       %[[RHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:       %[[LHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:       %[[RHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:       %[[LHS:.*]] = load { i64, i64 }, ptr %[[LHS_ADDR]]
// OGCG:       %[[RHS:.*]] = load { i64, i64 }, ptr %[[RHS_ADDR]]
// OGCG:       %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// OGCG:       %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// OGCG:       %[[PTR_CMP:.*]] = icmp eq i64 %[[LHS_PTR]], %[[RHS_PTR]]
// OGCG:       %[[PTR_NULL:.*]] = icmp eq i64 %[[LHS_PTR]], 0
// OGCG:       %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// OGCG:       %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// OGCG:       %[[ADJ_CMP:.*]] = icmp eq i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG-X86:   %[[TMP:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// OGCG-ARM:   %[[OR_ADJ:.*]] = or i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG-ARM:   %[[AND_ADJ:.*]] = and i64 %[[OR_ADJ]], 1
// OGCG-ARM:   %[[ADJ_CMP2:.*]] = icmp eq i64 %[[AND_ADJ]], 0
// OGCG-ARM:   %[[AND_PTR_NULL:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP2]]
// OGCG-ARM:   %[[TMP:.*]] = or i1 %[[AND_PTR_NULL]], %[[ADJ_CMP]]
// OGCG:       %[[RESULT:.*]] = and i1 %[[PTR_CMP]], %[[TMP]]

bool cmp_ne(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs != rhs;
}
  
// CIR-BEFORE: cir.func {{.*}} @_Z6cmp_neM3FooFviES1_
// CIR-BEFORE:   %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>
// CIR-BEFORE:   %[[CMP:.*]] = cir.cmp ne %[[LHS]], %[[RHS]] : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[CMP]], %{{.*}} : !cir.bool, !cir.ptr<!cir.bool>

// CIR-AFTER:     cir.func {{.*}} @_Z6cmp_neM3FooFviES1_
// CIR-AFTER:       %[[LHS:.*]] = cir.load{{.*}} %0 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:       %[[RHS:.*]] = cir.load{{.*}} %1 : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:       %[[NULL:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER:       %[[LHS_PTR:.*]] = cir.extract_member %[[LHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[RHS_PTR:.*]] = cir.extract_member %[[RHS]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[PTR_CMP:.*]] = cir.cmp ne %[[LHS_PTR]], %[[RHS_PTR]] : !s64i
// CIR-AFTER:       %[[PTR_NULL:.*]] = cir.cmp ne %[[LHS_PTR]], %[[NULL]] : !s64i
// CIR-AFTER:       %[[LHS_ADJ:.*]] = cir.extract_member %[[LHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[RHS_ADJ:.*]] = cir.extract_member %[[RHS]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:       %[[ADJ_CMP:.*]] = cir.cmp ne %[[LHS_ADJ]], %[[RHS_ADJ]] : !s64i
// CIR-AFTER-X86:   %[[TMP:.*]] = cir.and %[[PTR_NULL]], %[[ADJ_CMP]] : !cir.bool
// CIR-AFTER-ARM:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-AFTER-ARM:   %[[OR_ADJ:.*]] = cir.or %[[LHS_ADJ]], %[[RHS_ADJ]] : !s64i
// CIR-AFTER-ARM:   %[[AND_ADJ:.*]] = cir.and %[[OR_ADJ]], %[[ONE]] : !s64i
// CIR-AFTER-ARM:   %[[ADJ_CMP2:.*]] = cir.cmp ne %[[AND_ADJ]], %[[NULL]] : !s64i
// CIR-AFTER-ARM:   %[[AND_PTR_NULL:.*]] = cir.or %[[PTR_NULL]], %[[ADJ_CMP2]] : !cir.bool
// CIR-AFTER-ARM:   %[[TMP:.*]] = cir.and %[[AND_PTR_NULL]], %[[ADJ_CMP]] : !cir.bool
// CIR-AFTER:       %[[RESULT:.*]] = cir.or %[[PTR_CMP]], %[[TMP]] : !cir.bool

// LLVM:     define {{.*}} i1 @_Z6cmp_neM3FooFviES1_
// LLVM:       %[[LHS:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:       %[[RHS:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:       %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// LLVM:       %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// LLVM:       %[[PTR_CMP:.*]] = icmp ne i64 %[[LHS_PTR]], %[[RHS_PTR]]
// LLVM:       %[[PTR_NULL:.*]] = icmp ne i64 %[[LHS_PTR]], 0
// LLVM:       %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// LLVM:       %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// LLVM:       %[[ADJ_CMP:.*]] = icmp ne i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM-X86:   %[[TMP:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// LLVM-ARM:   %[[OR_ADJ:.*]] = or i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// LLVM-ARM:   %[[AND_ADJ:.*]] = and i64 %[[OR_ADJ]], 1
// LLVM-ARM:   %[[ADJ_CMP2:.*]] = icmp ne i64 %[[AND_ADJ]], 0
// LLVM-ARM:   %[[AND_PTR_NULL:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP2]]
// LLVM-ARM:   %[[TMP:.*]] = and i1 %[[AND_PTR_NULL]], %[[ADJ_CMP]]
// LLVM:       %[[RESULT:.*]] = or i1 %[[PTR_CMP]], %[[TMP]]

// OGCG:     define {{.*}} i1 @_Z6cmp_neM3FooFviES1_
// OGCG:       %[[LHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:       %[[RHS_TMP:.*]] = alloca { i64, i64 }
// OGCG:       %[[LHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:       %[[RHS_ADDR:.*]] = alloca { i64, i64 }
// OGCG:       %[[LHS:.*]] = load { i64, i64 }, ptr %[[LHS_ADDR]]
// OGCG:       %[[RHS:.*]] = load { i64, i64 }, ptr %[[RHS_ADDR]]
// OGCG:       %[[LHS_PTR:.*]] = extractvalue { i64, i64 } %[[LHS]], 0
// OGCG:       %[[RHS_PTR:.*]] = extractvalue { i64, i64 } %[[RHS]], 0
// OGCG:       %[[PTR_CMP:.*]] = icmp ne i64 %[[LHS_PTR]], %[[RHS_PTR]]
// OGCG:       %[[PTR_NULL:.*]] = icmp ne i64 %[[LHS_PTR]], 0
// OGCG:       %[[LHS_ADJ:.*]] = extractvalue { i64, i64 } %[[LHS]], 1
// OGCG:       %[[RHS_ADJ:.*]] = extractvalue { i64, i64 } %[[RHS]], 1
// OGCG:       %[[ADJ_CMP:.*]] = icmp ne i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG-X86:   %[[TMP:.*]] = and i1 %[[PTR_NULL]], %[[ADJ_CMP]]
// OGCG-ARM:   %[[OR_ADJ:.*]] = or i64 %[[LHS_ADJ]], %[[RHS_ADJ]]
// OGCG-ARM:   %[[AND_ADJ:.*]] = and i64 %[[OR_ADJ]], 1
// OGCG-ARM:   %[[ADJ_CMP2:.*]] = icmp ne i64 %[[AND_ADJ]], 0
// OGCG-ARM:   %[[AND_PTR_NULL:.*]] = or i1 %[[PTR_NULL]], %[[ADJ_CMP2]]
// OGCG-ARM:   %[[TMP:.*]] = and i1 %[[AND_PTR_NULL]], %[[ADJ_CMP]]
// OGCG:       %[[RESULT:.*]] = or i1 %[[PTR_CMP]], %[[TMP]]
