// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=constructor -dwarf-version=5 -O1 %s \
// RUN:            -o - | FileCheck %s -check-prefix CHECK-EDGES

// The following are identified edge cases involving the method being called:
// 1) Method is declared but not defined in current CU.
// 2) Pure virtual method but not defined in current CU.
// 3) Virtual method defined in a deeply nested structure hierarchy.

//---------------------------------------------------------------------
// 1) Method is declared but not defined in current CU - Pass.
//    Generate 'call_target' metadata for 'f1' and 'f2'.
//---------------------------------------------------------------------
struct CEmpty {
  virtual void f1();
  virtual void f2();
};

void CEmpty::f2() {
}

void edge_a(CEmpty *Empty) {
  Empty->f1();
  Empty->f2();
}

//---------------------------------------------------------------------
// 2) Pure virtual method but not defined in current CU - Pass.
//    Generate 'call_target' metadata for 'f1' and 'f2'.
//---------------------------------------------------------------------
struct CBase {
  virtual void f1() = 0;
  virtual void f2();
};

void CBase::f2() {
}

void edge_b(CBase *Base) {
  Base->f1();
  Base->f2();
}

//---------------------------------------------------------------------
// 3) Virtual method defined in a deeply nested structure hierarchy - Pass.
//    Generate 'call_target' metadata for 'd0', 'd1', 'd2' and 'd3'.
//---------------------------------------------------------------------
struct CD0 {
  struct CD1 {
    virtual void d1();
  };

  CD1 D1;
  virtual void d0();
};

void CD0::d0() {}
void CD0::CD1::d1() {}

void edge_c(CD0 *D0) {
  D0->d0();

  CD0::CD1 *D1 = &D0->D1;
  D1->d1();
}

// CHECK-EDGES: define {{.*}} @_Z6edge_aP6CEmpty{{.*}} {
// CHECK-EDGES:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[CEMPTY_F1_DCL:![0-9]+]]
// CHECK-EDGES:  call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[CEMPTY_F2_DCL:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES: define {{.*}} @_Z6edge_bP5CBase{{.*}} {
// CHECK-EDGES:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[CBASE_F1_DCL:![0-9]+]]
// CHECK-EDGES:  call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[CBASE_F2_DCL:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES: define {{.*}} @_Z6edge_cP3CD0{{.*}} {
// CHECK-EDGES:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[CD0_D0_DCL:![0-9]+]]
// CHECK-EDGES:  call void %4{{.*}} !dbg {{![0-9]+}}, !call_target [[CD0_D1_DCL:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES:  [[CD0_D1_DCL]] = {{.*}}!DISubprogram(name: "d1", linkageName: "_ZN3CD03CD12d1Ev", {{.*}}containingType
// CHECK-EDGES:  [[CD0_D0_DCL]] = {{.*}}!DISubprogram(name: "d0", linkageName: "_ZN3CD02d0Ev", {{.*}}containingType

// CHECK-EDGES:  [[CBASE_F1_DCL]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN5CBase2f1Ev", {{.*}}containingType
// CHECK-EDGES:  [[CBASE_F2_DCL]] = {{.*}}!DISubprogram(name: "f2", linkageName: "_ZN5CBase2f2Ev", {{.*}}containingType
// CHECK-EDGES:  [[CEMPTY_F2_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f2", linkageName: "_ZN6CEmpty2f2Ev", {{.*}}DISPFlagDefinition
// CHECK-EDGES:  [[CEMPTY_F2_DCL]] = {{.*}}!DISubprogram(name: "f2", linkageName: "_ZN6CEmpty2f2Ev", {{.*}}containingType
// CHECK-EDGES:  [[CEMPTY_F1_DCL]] = {{.*}}!DISubprogram(name: "f1", linkageName: "_ZN6CEmpty2f1Ev", {{.*}}containingType
// CHECK-EDGES:  [[CBASE_F2_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "f2", linkageName: "_ZN5CBase2f2Ev", {{.*}}DISPFlagDefinition

// CHECK-EDGES:  [[CD0_D0_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "d0", linkageName: "_ZN3CD02d0Ev", {{.*}}DISPFlagDefinition
// CHECK-EDGES:  [[CD0_D1_DEF:![0-9]+]] = {{.*}}!DISubprogram(name: "d1", linkageName: "_ZN3CD03CD12d1Ev", {{.*}}DISPFlagDefinition
