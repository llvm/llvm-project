// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

class A {
public:
  A();
};

void f() {
  static A a;
}

// CIR-BEFORE-LPP: cir.global "private" internal dso_local static_local @_ZZ1fvE1a = ctor : !rec_A {
// CIR-BEFORE-LPP:   %[[ADDR:.*]] = cir.get_global @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.call @_ZN1AC1Ev(%[[ADDR]]) : (!cir.ptr<!rec_A>) -> ()
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast}

// CIR-BEFORE-LPP: cir.func no_inline dso_local @_Z1fv()
// CIR-BEFORE-LPP:   %[[VAR:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.return

// OGCG: @_ZGVZ1fvE1a = internal global i64 0
// OGCG: define{{.*}}void @_Z1fv()
// OGCG:   %[[GUARD:.*]] = load atomic i8, ptr @_ZGVZ1fvE1a acquire
// OGCG:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// OGCG:   br i1 %[[IS_UNINIT]]
// OGCG: call i32 @__cxa_guard_acquire
// OGCG: call void @_ZN1AC1Ev
// OGCG: call void @__cxa_guard_release
// OGCG: ret void
