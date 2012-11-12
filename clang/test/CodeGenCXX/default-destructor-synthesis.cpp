// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -O2 -o - | FileCheck %s
static int count = 0;

struct S {
  S() { count++; }
  ~S() { count--; }
};

struct P {
  P() { count++; }
  ~P() { count--; }
};

struct Q {
  Q() { count++; }
  ~Q() { count--; }
};

struct M : Q, P {
  S s;
  Q q;
  P p;
  P p_arr[3];
  Q q_arr[2][3];
};
  
// CHECK: define i32 @_Z1fv() nounwind
int f() {
  {
    count = 1;
    M a;
  }

  // CHECK: ret i32 1
  return count;
}
