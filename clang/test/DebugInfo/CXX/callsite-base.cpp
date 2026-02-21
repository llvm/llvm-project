// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=standalone -dwarf-version=5 -O1 %s \
// RUN: -o - | FileCheck %s -check-prefix CHECK-BASE

// Simple class with only virtual methods: inlined and not-inlined
//
// The following three scenarios are considered:
// - out-of-line defined virtual member function (f1)
// - declared-but-not-defined virtual member function (f2)
// - inline defined virtual member function (f3)
//
// 1) We check for a generated 'call_target' for: 'f1', 'f2' and 'f3'.
// 2) Check that the 'CBase' type is defined.

struct CBase {
  virtual void f1();
  virtual void f2();
  virtual void f3() {}
};
void CBase::f1() {}

void bar(CBase *Base) {
  Base->f1();
  Base->f2();
  Base->f3();

  // Because this will instantiate the ctor, the CBase type should be defined.
  CBase B;
  B.f1();
}

// CHECK-BASE: %struct.CBase = type { ptr }

// CHECK-BASE: define {{.*}} @_Z3barP5CBase{{.*}} {
// CHECK-BASE:   alloca %struct.CBase
// CHECK-BASE:   call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_F1_DCL:![0-9]+]]
// CHECK-BASE:   call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_F2_DCL:![0-9]+]]
// CHECK-BASE:   call void %5{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_F3_DCL:![0-9]+]]
// CHECK-BASE:   call void @_ZN5CBaseC1Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-BASE:   call void @_ZN5CBase2f1Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-BASE: }

// CHECK-BASE: [[BASE_F1_DCL]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN5CBase2f1Ev", {{.*}}containingType
// CHECK-BASE: [[BASE_F2_DCL]] = {{.*}}!DISubprogram(name: "f2", linkageName: "_ZN5CBase2f2Ev", {{.*}}containingType
// CHECK-BASE: [[BASE_F3_DCL]] = {{.*}}!DISubprogram(name: "f3", linkageName: "_ZN5CBase2f3Ev", {{.*}}containingType

// CHECK-BASE: [[BASE_F1_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN5CBase2f1Ev", {{.*}}DISPFlagDefinition
// CHECK-BASE: [[BASE_F3_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f3", linkageName: "_ZN5CBase2f3Ev", {{.*}}DISPFlagDefinition
