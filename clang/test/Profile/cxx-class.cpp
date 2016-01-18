// Tests for instrumentation of C++ methods, constructors, and destructors.

// RUN: %clang %s -o - -emit-llvm -S -fprofile-instr-generate -fno-exceptions -target %itanium_abi_triple > %tgen
// RUN: FileCheck --input-file=%tgen -check-prefix=CTRGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=DTRGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=MTHGEN %s
// RUN: FileCheck --input-file=%tgen -check-prefix=WRPGEN %s

// RUN: llvm-profdata merge %S/Inputs/cxx-class.proftext -o %t.profdata
// RUN: %clang %s -o - -emit-llvm -S -fprofile-instr-use=%t.profdata -fno-exceptions -target %itanium_abi_triple > %tuse
// RUN: FileCheck --input-file=%tuse -check-prefix=CTRUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=DTRUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=MTHUSE %s
// RUN: FileCheck --input-file=%tuse -check-prefix=WRPUSE %s

class Simple {
  int Member;
public:
  // CTRGEN-LABEL: define {{.*}} @_ZN6SimpleC2Ei(
  // CTRUSE-LABEL: define {{.*}} @_ZN6SimpleC2Ei(
  // CTRGEN: store {{.*}} @[[SCC:__llvm_profile_counters__ZN6SimpleC2Ei]], i64 0, i64 0
  explicit Simple(int Member) : Member(Member) {
    // CTRGEN: store {{.*}} @[[SCC]], i64 0, i64 1
    // CTRUSE: br {{.*}} !prof ![[SC1:[0-9]+]]
    if (Member) {}
    // CTRGEN-NOT: store {{.*}} @[[SCC]],
    // CTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // CTRUSE: ret
  }
  // CTRUSE: ![[SC1]] = !{!"branch_weights", i32 100, i32 2}

  // DTRGEN-LABEL: define {{.*}} @_ZN6SimpleD2Ev(
  // DTRUSE-LABEL: define {{.*}} @_ZN6SimpleD2Ev(
  // DTRGEN: store {{.*}} @[[SDC:__llvm_profile_counters__ZN6SimpleD2Ev]], i64 0, i64 0
  ~Simple() {
    // DTRGEN: store {{.*}} @[[SDC]], i64 0, i64 1
    // DTRUSE: br {{.*}} !prof ![[SD1:[0-9]+]]
    if (Member) {}
    // DTRGEN-NOT: store {{.*}} @[[SDC]],
    // DTRUSE-NOT: br {{.*}} !prof ![0-9]+
    // DTRUSE: ret
  }
  // DTRUSE: ![[SD1]] = !{!"branch_weights", i32 100, i32 2}

  // MTHGEN-LABEL: define {{.*}} @_ZN6Simple6methodEv(
  // MTHUSE-LABEL: define {{.*}} @_ZN6Simple6methodEv(
  // MTHGEN: store {{.*}} @[[SMC:__llvm_profile_counters__ZN6Simple6methodEv]], i64 0, i64 0
  void method() {
    // MTHGEN: store {{.*}} @[[SMC]], i64 0, i64 1
    // MTHUSE: br {{.*}} !prof ![[SM1:[0-9]+]]
    if (Member) {}
    // MTHGEN-NOT: store {{.*}} @[[SMC]],
    // MTHUSE-NOT: br {{.*}} !prof ![0-9]+
    // MTHUSE: ret
  }
  // MTHUSE: ![[SM1]] = !{!"branch_weights", i32 100, i32 2}
};

// WRPGEN-LABEL: define {{.*}} @_Z14simple_wrapperv(
// WRPUSE-LABEL: define {{.*}} @_Z14simple_wrapperv(
// WRPGEN: store {{.*}} @[[SWC:__llvm_profile_counters__Z14simple_wrapperv]], i64 0, i64 0
void simple_wrapper() {
  // WRPGEN: store {{.*}} @[[SWC]], i64 0, i64 1
  // WRPUSE: br {{.*}} !prof ![[SW1:[0-9]+]]
  for (int I = 0; I < 100; ++I) {
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
