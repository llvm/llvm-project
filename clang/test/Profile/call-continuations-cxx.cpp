// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -emit-llvm -o - %s | FileCheck %s --check-prefix=IR

int init_value();

struct C {
  int x;
  C() : x(init_value()) {
    x = 2;
  }
};

int constructor_call(void) {
  C c;
  return c.x;
}

// IR-DAG: @__profc__Z16constructor_callv = private global [2 x i64]
// IR-DAG: @__profc__ZN1CC2Ev = linkonce_odr hidden global [2 x i64]

// IR-LABEL: define{{.*}} i32 @_Z16constructor_callv(
// IR: call void @_ZN1CC1Ev
// IR-NEXT: load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc__Z16constructor_callv, i32 0, i32 1)

// IR-LABEL: define{{.*}} void @_ZN1CC2Ev(
// IR: call{{.*}} @_Z10init_valuev
// IR-NEXT: load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc__ZN1CC2Ev, i32 0, i32 1)
