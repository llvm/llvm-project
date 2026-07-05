// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -emit-llvm -o - -std=c++11 | FileCheck %s
namespace std {
struct type_info;
}

struct A {
  virtual ~A();
  operator bool();
};
struct B : A {};

void f1(A *x) { typeid(false, *x); }
// CHECK-LABEL: define {{.*}}void @_Z2f1P1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f2(bool b, A *x, A *y) { typeid(b ? *x : *y); }
// CHECK-LABEL: define {{.*}}void @_Z2f2bP1AS0_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f3(bool b, A *x, A &y) { typeid(b ? *x : y); }
// CHECK-LABEL: define {{.*}}void @_Z2f3bP1ARS_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f4(bool b, A &x, A *y) { typeid(b ? x : *y); }
// CHECK-LABEL: define {{.*}}void @_Z2f4bR1APS_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f5(volatile A *x) { typeid(*x); }
// CHECK-LABEL: define {{.*}}void @_Z2f5PV1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f6(A *x) { typeid((B &)*(B *)x); }
// CHECK-LABEL: define {{.*}}void @_Z2f6P1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f7(A *x) { typeid((*x)); }
// CHECK-LABEL: define {{.*}}void @_Z2f7P1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f8(A *x) { typeid(x[0]); }
// CHECK-LABEL: define {{.*}}void @_Z2f8P1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f9(A *x) { typeid(0[x]); }
// CHECK-LABEL: define {{.*}}void @_Z2f9P1A
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f10(A *x, A *y) { typeid(*y ?: *x); }
// CHECK-LABEL: define {{.*}}void @_Z3f10P1AS0_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f11(A *x, A &y) { typeid(*x ?: y); }
// CHECK-LABEL: define {{.*}}void @_Z3f11P1ARS_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f12(A &x, A *y) { typeid(x ?: *y); }
// CHECK-LABEL: define {{.*}}void @_Z3f12R1APS_
// CHECK:       icmp eq {{.*}}, null
// CHECK-NEXT:  br i1

void f13(A &x, A &y) { typeid(x ?: y); }
// CHECK-LABEL: define {{.*}}void @_Z3f13R1AS0_
// CHECK-NOT:   icmp eq {{.*}}, null

void f14(A *x) { typeid((const A &)(A)*x); }
// CHECK-LABEL: define {{.*}}void @_Z3f14P1A
// CHECK-NOT:   icmp eq {{.*}}, null

void f15(A *x) { typeid((A &&)*(A *)nullptr); }
// CHECK-LABEL: define {{.*}}void @_Z3f15P1A
// CHECK-NOT:   icmp eq {{.*}}, null
