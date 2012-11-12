// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
struct X {
  X();
  ~X();
};

struct Y {
  Y();
  ~Y();
};

// CHECK: define void @_Z1fiPPKc(
void f(int argc, const char* argv[]) {
  // CHECK: call void @_ZN1XC1Ev
  X x;
  // CHECK: call i8* @llvm.stacksave(
  const char *argv2[argc];
  // CHECK: call void @_ZN1YC1Ev
  Y y;
  for (int i = 0; i != argc; ++i)
    argv2[i] = argv[i];

  // CHECK: call void @_ZN1YD1Ev
  // CHECK: call void @llvm.stackrestore
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: ret void
}

namespace PR11744 {
  // Make sure this doesn't crash; there was a use-after-free issue
  // for this testcase.
  template<typename T> int f(int n) {
    T arr[3][n];
    return 3;
  }
  int test = f<int>(0);
}
