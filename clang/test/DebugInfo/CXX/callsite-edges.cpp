// Check edge cases:

//---------------------------------------------------------------------
// Method is declared but not defined in current CU - Fail.
// No debug information entry is generated for 'one'.
// Generate 'call_target' metadata only for 'two'.
//---------------------------------------------------------------------
class CEmpty {
public:
  virtual void one(bool Flag);
  virtual void two(int P1, char P2);
};

void CEmpty::two(int P1, char P2) {
}

void edge_a(CEmpty *Empty) {
  Empty->one(false);
  Empty->two(77, 'a');
}

//---------------------------------------------------------------------
// Pure virtual method but not defined in current CU - Pass.
// Generate 'call_target' metadata for 'one' and 'two'.
//---------------------------------------------------------------------
class CBase {
public:
  virtual void one(bool Flag) = 0;
  virtual void two(int P1, char P2);
};

void CBase::two(int P1, char P2) {
}

void edge_b(CBase *Base) {
  Base->one(false);
  Base->two(77, 'a');
}

//---------------------------------------------------------------------
// Virtual method defined very deeply - Pass.
// Generate 'call_target' metadata for 'd0', 'd1', 'd2' and 'd3'.
//---------------------------------------------------------------------
struct CDeep {
  struct CD1 {
    struct CD2 {
      struct CD3 {
        virtual void d3(int P3);
      };

      CD3 D3;
      virtual void d2(int P2);
    };

    CD2 D2;
    virtual void d1(int P1);
  };

  CD1 D1;
  virtual void d0(int P);
};

void CDeep::d0(int P) {}
void CDeep::CD1::d1(int P1) {}
void CDeep::CD1::CD2::d2(int P2) {}
void CDeep::CD1::CD2::CD3::d3(int P3) {}

void edge_c(CDeep *Deep) {
  Deep->d0(0);

  CDeep::CD1 *D1 = &Deep->D1;
  D1->d1(1);

  CDeep::CD1::CD2 *D2 = &D1->D2;
  D2->d2(2);

  CDeep::CD1::CD2::CD3 *D3 = &D2->D3;
  D3->d3(3);
}

// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=constructor -dwarf-version=5 -O1 %s \
// RUN:            -o - | FileCheck %s -check-prefix CHECK-EDGES

// CHECK-EDGES: define {{.*}} @_Z6edge_aP6CEmpty{{.*}} {
// CHECK-EDGES-DAG:  call void %1{{.*}} !dbg {{![0-9]+}}
// CHECK-EDGES-DAG:  call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[CEMPTY_TWO:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES: define {{.*}} @_Z6edge_bP5CBase{{.*}} {
// CHECK-EDGES-DAG:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[CBASE_ONE:![0-9]+]]
// CHECK-EDGES-DAG:  call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[CBASE_TWO:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES: define {{.*}} @_Z6edge_cP5CDeep{{.*}} {
// CHECK-EDGES-DAG:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[CDEEP_D0:![0-9]+]]
// CHECK-EDGES-DAG:  call void %4{{.*}} !dbg {{![0-9]+}}, !call_target [[CDEEP_D1:![0-9]+]]
// CHECK-EDGES-DAG:  call void %7{{.*}} !dbg {{![0-9]+}}, !call_target [[CDEEP_D2:![0-9]+]]
// CHECK-EDGES-DAG:  call void %10{{.*}} !dbg {{![0-9]+}}, !call_target [[CDEEP_D3:![0-9]+]]
// CHECK-EDGES: }

// CHECK-EDGES-DAG:  [[CEMPTY_TWO]] = {{.*}}!DISubprogram(name: "two", linkageName: "_ZN6CEmpty3twoEic"
// CHECK-EDGES-DAG:  [[CBASE_ONE]] = {{.*}}!DISubprogram(name: "one", linkageName: "_ZN5CBase3oneEb"
// CHECK-EDGES-DAG:  [[CBASE_TWO]] = {{.*}}!DISubprogram(name: "two", linkageName: "_ZN5CBase3twoEic"

// CHECK-EDGES-DAG:  [[CDEEP_D0]] = {{.*}}!DISubprogram(name: "d0", linkageName: "_ZN5CDeep2d0Ei"
// CHECK-EDGES-DAG:  [[CDEEP_D1]] = {{.*}}!DISubprogram(name: "d1", linkageName: "_ZN5CDeep3CD12d1Ei"
// CHECK-EDGES-DAG:  [[CDEEP_D2]] = {{.*}}!DISubprogram(name: "d2", linkageName: "_ZN5CDeep3CD13CD22d2Ei"
// CHECK-EDGES-DAG:  [[CDEEP_D3]] = {{.*}}!DISubprogram(name: "d3", linkageName: "_ZN5CDeep3CD13CD23CD32d3Ei"
