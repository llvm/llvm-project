// Tests for instrumentation of C++ methods, constructors, and destructors.

// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=clang -triple %itanium_abi_triple > %tgen
// RUN: FileCheck --input-file=%tgen -check-prefix=CTRGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=DTRGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=MTHGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=WRPGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=VCTRGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=VDTRGEN %s

// RUN: llvm-profdata merge %S/Inputs/cxx-class.proftext -o %t.profdata
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -triple %itanium_abi_triple > %tuse
// RUN: FileCheck --input-file=%tuse -check-prefix=CTRUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=DTRUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=MTHUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=WRPUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=VCTRUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=VDTRUSE %s

class Simple {
public:
  int Member;
  // CTRGEN-LABEL: define {{.*}} @_ZN6SimpleC2Ei(
  // CTRUSE-LABEL: define {{.*}} @_ZN6SimpleC2Ei(
  // CTRGEN: store {{.*}} @[[SCC:__profc__ZN6SimpleC2Ei]]
  explicit Simple(int Member) : Member(Member) {
    // CTRGEN: store {{.*}} @[[SCC]], i32 0, i32 1
    // CTRUSE: br {{.*}} !prof ![[SC1:[0-9]+]]
    if (Member) {}
    // CTRGEN-NOT: store {{.*}} @[[SCC]],
    // CTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // CTRUSE: ret
  }
  // CTRUSE: ![[SC1]] = !{!"branch_weights", i32 100, i32 2}

  // DTRGEN-LABEL: define {{.*}} @_ZN6SimpleD2Ev(
  // DTRUSE-LABEL: define {{.*}} @_ZN6SimpleD2Ev(
  // DTRGEN: store {{.*}} @[[SDC:__profc__ZN6SimpleD2Ev]]
  ~Simple() {
    // DTRGEN: store {{.*}} @[[SDC]], i32 0, i32 1
    // DTRUSE: br {{.*}} !prof ![[SD1:[0-9]+]]
    if (Member) {}
    // DTRGEN-NOT: store {{.*}} @[[SDC]],
    // DTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // DTRUSE: ret
  }
  // DTRUSE: ![[SD1]] = !{!"branch_weights", i32 100, i32 2}

  // MTHGEN-LABEL: define {{.*}} @_ZN6Simple6methodEv(
  // MTHUSE-LABEL: define {{.*}} @_ZN6Simple6methodEv(
  // MTHGEN: store {{.*}} @[[SMC:__profc__ZN6Simple6methodEv]]
  void method() {
    // MTHGEN: store {{.*}} @[[SMC]], i32 0, i32 1
    // MTHUSE: br {{.*}} !prof ![[SM1:[0-9]+]]
    if (Member) {}
    // MTHGEN-NOT: store {{.*}} @[[SMC]],
    // MTHUSE-NOT: br {{.*}} !prof ![0-9]+
    // MTHUSE: ret
  }
  // MTHUSE: ![[SM1]] = !{!"branch_weights", i32 100, i32 2}
};

class Derived : virtual public Simple {
public:
  // VCTRGEN-LABEL: define {{.*}} @_ZN7DerivedC1Ev(
  // VCTRUSE-LABEL: define {{.*}} @_ZN7DerivedC1Ev(
  // VCTRGEN: store {{.*}} @[[SCC:__profc__ZN7DerivedC1Ev]]
  Derived() : Simple(0) {
    // VCTRGEN: store {{.*}} @[[SCC]], i32 0, i32 1
    // VCTRUSE: br {{.*}} !prof ![[SC1:[0-9]+]]
    if (Member) {}
    // VCTRGEN-NOT: store {{.*}} @[[SCC]],
    // VCTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // VCTRUSE: ret
  }
  // VCTRUSE: ![[SC1]] = !{!"branch_weights", i32 100, i32 2}

  // VDTRGEN-LABEL: define {{.*}} @_ZN7DerivedD2Ev(
  // VDTRUSE-LABEL: define {{.*}} @_ZN7DerivedD2Ev(
  // VDTRGEN: store {{.*}} @[[SDC:__profc__ZN7DerivedD2Ev]]
  ~Derived() {
    // VDTRGEN: store {{.*}} @[[SDC]], i32 0, i32 1
    // VDTRUSE: br {{.*}} !prof ![[SD1:[0-9]+]]
    if (Member) {}
    // VDTRGEN-NOT: store {{.*}} @[[SDC]],
    // VDTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // VDTRUSE: ret
  }
  // VDTRUSE: ![[SD1]] = !{!"branch_weights", i32 100, i32 2}
};

// WRPGEN-LABEL: define {{.*}} @_Z14simple_wrapperv(
// WRPUSE-LABEL: define {{.*}} @_Z14simple_wrapperv(
// WRPGEN: store {{.*}} @[[SWC:__profc__Z14simple_wrapperv]]
void simple_wrapper() {
  // WRPGEN: store {{.*}} @[[SWC]], i32 0, i32 1
  // WRPUSE: br {{.*}} !prof ![[SW1:[0-9]+]]
  for (int I = 0; I < 100; ++I) {
    Derived d;
    Simple S(I);
    S.method();
  }
  // WRPGEN-NOT: store {{.*}} @[[SWC]],
  // WRPUSE-NOT: br {{.*}} !prof ![0-9]+
  // WRPUSE: ret
}
// WRPUSE: ![[SW1]] = !{!"branch_weights", i32 101, i32 2}

int main(int argc, const char *argv[]) {
  simple_wrapper();
  return 0;
}
