// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir -o - %s | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm -o - %s | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s -check-prefix=OGCG

// CIR: !rec_A = !cir.record<class "A" {!s32i} #cir.record.decl.ast>
// CIR: cir.global "private" constant external @_ZN1B1AE : !cir.ptr<!rec_A> {alignment = 8 : i64}

// LLVM: @_ZN1B1AE = external local_unnamed_addr constant ptr, align 8
// OGCG: @_ZN1B1AE = external local_unnamed_addr constant ptr, align 8
class A { int p = 1;};
class B {
public:
  static A &A;
};
A& ref() {
  // CIR-LABEL: _Z3refv
  // CIR: [[GLOBAL:%.*]] = cir.get_global @_ZN1B1AE : !cir.ptr<!cir.ptr<!rec_A>>
  // CIR: [[LD1:%.*]] = cir.load [[GLOBAL]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR: cir.store{{.*}} [[LD1]], [[ALLOCA:%.*]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
  // CIR: [[LD2:%.*]] = cir.load{{.*}} [[ALLOCA:%.*]]: !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR: cir.return [[LD2]] : !cir.ptr<!rec_A>

  // LLVM-LABEL: _Z3refv
  // LLVM: [[LD:%.*]] = load ptr, ptr @_ZN1B1AE
  // LLVM-NEXT: ret ptr [[LD]]

  // OGCG-LABEL: _Z3refv
  // OGCG: [[LD:%.*]] = load ptr, ptr @_ZN1B1AE
  // OGCG-NEXT: ret ptr [[LD]]
  return B::A;
}
