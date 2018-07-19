// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -std=c++11 -fmodules-cache-path=%t -I %S/Inputs -triple %itanium_abi_triple -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -std=c++11 -fmodules-cache-path=%t -I %S/Inputs -triple %itanium_abi_triple -disable-llvm-passes -emit-llvm -debug-info-kind=limited -o - %s | FileCheck %s
// FIXME: When we have a syntax for modules in C++, use that.

@import cxx_irgen_top;

CtorInit<int> x;

@import cxx_irgen_left;
@import cxx_irgen_right;

// Keep these two namespace definitions separate; merging them hides the bug.
namespace EmitInlineMethods {
  // CHECK-DAG: define linkonce_odr [[CC:([0-9_a-z]*cc[ ]+)?]]void @_ZN17EmitInlineMethods1C1fEPNS_1AE(
  // CHECK-DAG: declare [[CC]]void @_ZN17EmitInlineMethods1A1gEv(
  struct C {
    __attribute__((used)) void f(A *p) { p->g(); }
  };
}
namespace EmitInlineMethods {
  // CHECK-DAG: define linkonce_odr [[CC]]void @_ZN17EmitInlineMethods1D1fEPNS_1BE(
  // CHECK-DAG: define linkonce_odr [[CC]]void @_ZN17EmitInlineMethods1B1gEv(
  struct D {
    __attribute__((used)) void f(B *p) { p->g(); }
  };
}

// CHECK-DAG: define available_externally hidden {{.*}}{{signext i32|i32}} @_ZN1SIiE1gEv({{.*}} #[[ALWAYS_INLINE:.*]] align
int a = S<int>::g();

int b = h();

// CHECK-DAG: define linkonce_odr {{.*}}{{signext i32|i32}} @_Z3minIiET_S0_S0_(i32
int c = min(1, 2);
// CHECK: define available_externally {{.*}}{{signext i32|i32}} @_ZN1SIiE1fEv({{.*}} #[[ALWAYS_INLINE]] align

namespace ImplicitSpecialMembers {
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1BC2ERKS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1BC2EOS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1CC2ERKS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1CC2EOS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1DC2ERKS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN22ImplicitSpecialMembers1DC2EOS0_(
  // CHECK: call {{.*}} @_ZN22ImplicitSpecialMembers1AC1ERKS0_(
  // CHECK-LABEL: define {{.*}} @_ZN20OperatorDeleteLookup1AD0Ev(
  // CHECK: call {{.*}}void @_ZN20OperatorDeleteLookup1AdlEPv(

  // CHECK-DAG: call {{[a-z\_\d]*[ ]?i32}} @_ZN8CtorInitIiE1fEv(

  extern B b1;
  B b2(b1);
  B b3(static_cast<B&&>(b1));

  extern C c1;
  C c2(c1);
  C c3(static_cast<C&&>(c1));

  extern D d1;
  D d2(d1);
  D d3(static_cast<D&&>(d1));
}

namespace OperatorDeleteLookup {
  // Trigger emission of B's vtable and deleting dtor.
  // This requires us to know what operator delete was selected.
  void g() { f(); }
}


// CHECK: attributes #[[ALWAYS_INLINE]] = {{.*}} alwaysinline
