// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Empty { };
struct A { 
};

struct B : A, Empty { 
  B() : A(), Empty() { }
};

void f() {
  B b1;
}

// CHECK-LABEL: @_ZN1BC2Ev
// CHECK: %[[A:.*]] = cir.base_class_addr({{.*}}) [0] -> !cir.ptr<!ty_A>
// CHECK: cir.call @_ZN1AC2Ev(%[[A:.*]]) : (!cir.ptr<!ty_A>) -> ()
// CHECK: %[[BASE:.*]] = cir.base_class_addr({{.*}}) [0] -> !cir.ptr<!ty_Empty>
// CHECK: cir.call @_ZN5EmptyC2Ev(%[[BASE]]) : (!cir.ptr<!ty_Empty>) -> ()