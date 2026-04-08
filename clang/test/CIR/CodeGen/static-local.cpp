// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

void use_static_decl() {
  static int x = 42;
  static int *p = &x;
}

// CIR-DAG: cir.global "private" internal dso_local @_ZZ15use_static_declvE1p = #cir.global_view<@_ZZ15use_static_declvE1x> : !cir.ptr<!s32i>
// CIR-DAG: cir.global "private" internal dso_local @_ZZ15use_static_declvE1x = #cir.int<42> : !s32i

// LLVM-DAG: @_ZZ15use_static_declvE1p = internal global ptr @_ZZ15use_static_declvE1x
// LLVM-DAG: @_ZZ15use_static_declvE1x = internal global i32 42

// OGCG-DAG: @_ZZ15use_static_declvE1x = internal global i32 42
// OGCG-DAG: @_ZZ15use_static_declvE1p = internal global ptr @_ZZ15use_static_declvE1x

class A {
public:
  A();
};

void use(A*);
void f() {
  static A a;
  use(&a);
}

// CIR-BEFORE-LPP: cir.global "private" internal dso_local static_local_guard<"_ZGVZ1fvE1a"> @_ZZ1fvE1a = ctor : !rec_A {
// CIR-BEFORE-LPP:   %[[ADDR:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.call @_ZN1AC1Ev(%[[ADDR]]) : (!cir.ptr<!rec_A> {{.*}}) -> ()
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast}

// CIR-BEFORE-LPP: cir.func no_inline dso_local @_Z1fv()
// CIR-BEFORE-LPP:   %[[VAR:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.call @_Z3useP1A(%[[VAR]])
// CIR-BEFORE-LPP:   cir.return

// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ1fvE1a = #cir.int<0> : !s64i
// CIR: cir.func{{.*}}@_Z1fv()
// CIR:   %[[ADDR:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR:   %[[GUARD:.*]] = cir.get_global @_ZGVZ1fvE1a : !cir.ptr<!s64i>
// CIR:   %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:   %[[GUARD_LOAD:.*]] = cir.load{{.*}}%[[GUARD_BYTE_PTR]]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0>
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]]
// CIR:   cir.if %[[IS_UNINIT]]
// CIR:     cir.call @__cxa_guard_acquire
// CIR:     cir.if
// CIR:       cir.call @_ZN1AC1Ev
// CIR:       cir.call @__cxa_guard_release
// CIR:   cir.call @_Z3useP1A(%[[ADDR]])
// CIR:   cir.return

// LLVM-DAG: @_ZGVZ1fvE1a = internal global i64 0
// LLVM: define{{.*}}void @_Z1fv()
// LLVM:   %[[GUARD:.*]] = load atomic i8, ptr @_ZGVZ1fvE1a acquire
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// LLVM: call i32 @__cxa_guard_acquire
// LLVM: call void @_ZN1AC1Ev
// LLVM: call void @__cxa_guard_release
// LLVM: call void @_Z3useP1A(ptr {{.*}}@_ZZ1fvE1a)
// LLVM: ret void

// OGCG-DAG: @_ZGVZ1fvE1a = internal global i64 0
// OGCG: define{{.*}}void @_Z1fv()
// OGCG:   %[[GUARD:.*]] = load atomic i8, ptr @_ZGVZ1fvE1a acquire
// OGCG:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// OGCG:   br i1 %[[IS_UNINIT]]
// OGCG: call i32 @__cxa_guard_acquire
// OGCG: call void @_ZN1AC1Ev
// OGCG: call void @__cxa_guard_release
// OGCG: call void @_Z3useP1A(ptr {{.*}}@_ZZ1fvE1a)
// OGCG: ret void
