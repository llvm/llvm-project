// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -O3 -fno-experimental-new-pass-manager  -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LEGACY
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -O3 -fexperimental-new-pass-manager  -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NEWPM
// RUN: %clang_cc1 %s -triple=x86_64-windows-gnu -emit-llvm -o - | FileCheck %s -check-prefix MINGW64
struct A {
  virtual int vf1() { return 1; }
  virtual int vf2() { return 2; }
};

int f(A* a, int (A::*fp)()) {
  return (a->*fp)();
}

// CHECK-LABEL: define i32 @_Z2g1v()
// CHECK-LEGACY: ret i32 1
// CHECK-NEWPM: [[A:%.*]] = alloca %struct.A, align 8
// CHECK-NEWPM: [[TMP:%.*]] = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 0
// CHECK-NEWPM: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** [[TMP]], align 8
// CHECK-NEWPM: [[RET:%.*]] = call i32 @_ZN1A3vf1Ev(%struct.A* nonnull %a) #2
// CHECK-NEWPM: ret i32 [[RET]]
// MINGW64-LABEL: define dso_local i32 @_Z2g1v()
// MINGW64: call i32 @_Z1fP1AMS_FivE(%struct.A* %{{.*}}, { i64, i64 }* %{{.*}})
int g1() {
  A a;
  return f(&a, &A::vf1);
}

// CHECK-LABEL: define i32 @_Z2g2v()
// CHECK: ret i32 2
// MINGW64-LABEL: define dso_local i32 @_Z2g2v()
// MINGW64: call i32 @_Z1fP1AMS_FivE(%struct.A* %{{.*}}, { i64, i64 }* %{{.*}})
int g2() {
  A a;
  return f(&a, &A::vf2);
}
