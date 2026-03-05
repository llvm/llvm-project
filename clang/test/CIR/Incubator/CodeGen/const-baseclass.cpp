// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

struct Empty { };
struct A {
};

struct B : A, Empty {
  B() : A(), Empty() { }
};

void f() {
  B b1;
}

// Trivial base class constructor calls are lowered away.
// CHECK-LABEL: @_ZN1BC2Ev
// CHECK: %[[A:.*]] = cir.base_class_addr {{.*}} [0] -> !cir.ptr<!rec_A>
// CHECK: %[[BASE:.*]] = cir.base_class_addr {{.*}} [0] -> !cir.ptr<!rec_Empty>
// CHECK: cir.return

// LLVM-LABEL: define {{.*}} @_ZN1BC2Ev
// LLVM-NOT:     call {{.*}} @_ZN1AC2Ev
// LLVM-NOT:     call {{.*}} @_ZN5EmptyC2Ev
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} @_ZN1BC2Ev
// OGCG-NOT:     call {{.*}} @_ZN1AC2Ev
// OGCG-NOT:     call {{.*}} @_ZN5EmptyC2Ev
// OGCG:         ret void
