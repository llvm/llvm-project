// RUN: %clang_cc1 %s -O2 -emit-llvm -fextend-lifetimes -o - | FileCheck %s
// Check that fake use calls are emitted at the correct locations, i.e.
// at the end of lexical blocks and at the end of the function.

extern int use(int);
int glob1;
int glob2;
float globf;

int foo(int i) {
  // CHECK: define{{.*}}foo
  if (i < 4) {
    int j = i * 3;
    if (glob1 > 3) {
      float f = globf;
      // CHECK: [[SSAVAL:%[a-z0-9]*]] = load float{{.*}}globf
      j = f;
      glob2 = j;
      // CHECK: store{{.*}}glob2
      // CHECK-NEXT: call void (...) @llvm.fake.use(float [[SSAVAL]])
    }
    glob1 = j;
    // CHECK: store{{.*}}glob1
    // CHECK-NEXT: call void (...) @llvm.fake.use(i32 %j.
  }
  // CHECK: call void (...) @llvm.fake.use(i32 %i)
  // CHECK-NEXT: ret
  return 4;
}
