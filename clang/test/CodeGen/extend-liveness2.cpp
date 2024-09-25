// RUN: %clang_cc1 %s -O2 -emit-llvm -fextend-lifetimes -fcxx-exceptions -fexceptions -o - | FileCheck %s
// REQUIRES: x86-registered-target
// This test checks that the fake_use concept works with exception handling and that we
// can handle the __int128 data type.

class A {
public:
  A(int i) : m_i(i) {}
  void func(__int128 i128);

  int m_i;
};

extern int bar();
extern void foo();
int glob;

void A::func(__int128 i128) {
  int j = 4;
  try {
    int k = bar();
    foo();
    // CHECK: [[SSAVAL:%[a-z0-9]*]] = invoke{{.*}}bar
    glob = 0;
    // CHECK: store{{.*}}glob
    // CHECK-NEXT: call void (...) @llvm.fake.use(i32 [[SSAVAL]])
  } catch (...) {
    foo();
  }
  // CHECK-LABEL: try.cont:
  // CHECK-DAG: call void (...) @llvm.fake.use({{.*%this}})
  // CHECK-DAG: call void (...) @llvm.fake.use(i128 %i128.sroa.0.0.insert.insert)
  // CHECK: ret void
}
