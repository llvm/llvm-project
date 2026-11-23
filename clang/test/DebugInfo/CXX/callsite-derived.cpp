// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=constructor -dwarf-version=5 -O1 %s \
// RUN:            -o - | FileCheck %s -check-prefix CHECK-DERIVED

// Simple base and derived class with virtual and static methods:
// We check for:
// - a generated 'call_target' for 'f1'.
// - not generated 'call_target' for 'f3'.

struct CBase {
  virtual void f1() {}
  static void f3();
};

void CBase::f3() {
}

void foo(CBase *Base) {
  CBase::f3();
}

struct CDerived : public CBase {
  void f1() {}
};
void foo(CDerived *Derived);

int main() {
  CDerived D;
  foo(&D);

  return 0;
}

void foo(CDerived *Derived) {
  Derived->f1();
}

// CHECK-DERIVED: define {{.*}} @_Z3fooP5CBase{{.*}} {
// CHECK-DERIVED: call void @_ZN5CBase2f3Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @main{{.*}} {
// CHECK-DERIVED:  call void @_ZN8CDerivedC1Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED:  call void @_Z3fooP8CDerived{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_ZN8CDerivedC1Ev{{.*}} {
// CHECK-DERIVED:  call void @_ZN8CDerivedC2Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_Z3fooP8CDerived{{.*}} {
// CHECK-DERIVED:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[DERIVED_F1_DCL:![0-9]+]]
// CHECK-DERIVED: }

// CHECK-DERIVED: [[BASE_F1_DCL:![0-9]+]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN5CBase2f1Ev", {{.*}}containingType
// CHECK-DERIVED: [[DERIVED_F1_DCL]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN8CDerived2f1Ev", {{.*}}containingType
// CHECK-DERIVED: [[DERIVED_F1_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN8CDerived2f1Ev", {{.*}}DISPFlagDefinition
// CHECK-DERIVED: [[BASE_F1_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN5CBase2f1Ev", {{.*}}DISPFlagDefinition
