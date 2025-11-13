// Simple base and derived class with virtual and static methods:
// We check for a generated 'call_target' for:
// - 'one', 'two' and 'three'.

class CBase {
public:
  virtual void one(bool Flag) {}
  virtual void two(int P1, char P2) {}
  static void three();
};

void CBase::three() {
}
void bar(CBase *Base);

void foo(CBase *Base) {
  CBase::three();
}

class CDerived : public CBase {
public:
  void one(bool Flag) {}
  void two(int P1, char P2) {}
};
void foo(CDerived *Derived);

int main() {
  CBase B;
  bar(&B);

  CDerived D;
  foo(&D);

  return 0;
}

void bar(CBase *Base) {
  Base->two(77, 'a');
}

void foo(CDerived *Derived) {
  Derived->one(true);
}

// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=constructor -dwarf-version=5 -O1 %s \
// RUN:            -o - | FileCheck %s -check-prefix CHECK-DERIVED

// CHECK-DERIVED: define {{.*}} @_Z3fooP5CBase{{.*}} {
// CHECK-DERIVED-DAG: call void @_ZN5CBase5threeEv{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @main{{.*}} {
// CHECK-DERIVED-DAG:  call void @_ZN5CBaseC1Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED-DAG:  call void @_Z3barP5CBase{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED-DAG:  call void @_ZN8CDerivedC1Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED-DAG:  call void @_Z3fooP8CDerived{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_ZN5CBaseC1Ev{{.*}} {
// CHECK-DERIVED-DAG:  call void @_ZN5CBaseC2Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_Z3barP5CBase{{.*}} {
// CHECK-DERIVED-DAG:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_TWO:![0-9]+]]
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_ZN8CDerivedC1Ev{{.*}} {
// CHECK-DERIVED-DAG:  call void @_ZN8CDerivedC2Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-DERIVED: }

// CHECK-DERIVED: define {{.*}} @_Z3fooP8CDerived{{.*}} {
// CHECK-DERIVED-DAG:  call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[DERIVED_ONE:![0-9]+]]
// CHECK-DERIVED: }

// CHECK-DERIVED-DAG: [[BASE_TWO]] = {{.*}}!DISubprogram(name: "two", linkageName: "_ZN5CBase3twoEic"
// CHECK-DERIVED-DAG: [[DERIVED_ONE]] = {{.*}}!DISubprogram(name: "one", linkageName: "_ZN8CDerived3oneEb"
