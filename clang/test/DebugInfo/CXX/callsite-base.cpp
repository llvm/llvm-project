// Simple class with only virtual methods: inlined and not-inlined
// We check for a generated 'call_target' for:
// - 'one', 'two' and 'three'.

class CBase {
public:
  virtual void one();
  virtual void two();
  virtual void three() {}
};
void CBase::one() {}

void bar(CBase *Base) {
  Base->one();
  Base->two();
  Base->three();

  CBase B;
  B.one();
}

// RUN: %clang_cc1 -triple=x86_64-linux -disable-llvm-passes -emit-llvm \
// RUN:            -debug-info-kind=constructor -dwarf-version=5 -O1 %s \
// RUN: -o - | FileCheck %s -check-prefix CHECK-BASE

// CHECK-BASE: define {{.*}} @_Z3barP5CBase{{.*}} {
// CHECK-BASE-DAG:   call void %1{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_ONE:![0-9]+]]
// CHECK-BASE-DAG:   call void %3{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_TWO:![0-9]+]]
// CHECK-BASE-DAG:   call void %5{{.*}} !dbg {{![0-9]+}}, !call_target [[BASE_THREE:![0-9]+]]
// CHECK-BASE-DAG:   call void @_ZN5CBaseC2Ev{{.*}} !dbg {{![0-9]+}}
// CHECK-BASE-DAG:   call void @_ZN5CBase3oneEv{{.*}} !dbg {{![0-9]+}}
// CHECK-BASE: }

// CHECK-BASE-DAG: [[BASE_ONE]] = {{.*}}!DISubprogram(name: "one", linkageName: "_ZN5CBase3oneEv"
// CHECK-BASE-DAG: [[BASE_TWO]] = {{.*}}!DISubprogram(name: "two", linkageName: "_ZN5CBase3twoEv"
// CHECK-BASE-DAG: [[BASE_THREE]] = {{.*}}!DISubprogram(name: "three", linkageName: "_ZN5CBase5threeEv"
