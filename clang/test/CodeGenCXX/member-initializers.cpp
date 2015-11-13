// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -O3 | FileCheck %s

struct A {
  virtual int f() { return 1; }
};

struct B : A {
  B() : i(f()) { }
  
  virtual int f() { return 2; }
  
  int i;
};

// CHECK-LABEL: define i32 @_Z1fv() #0
int f() {
  B b;
  
  // CHECK: ret i32 2
  return b.i;
}

// Test that we don't try to fold the default value of j when initializing i.
// CHECK: define i32 @_Z9test_foldv() [[NUW_RN:#[0-9]+]]
int test_fold() {
  struct A {
    A(const int j = 1) : i(j) { } 
    int i;
  };

  // CHECK: ret i32 2
  return A(2).i;
}

// CHECK: attributes [[NUW_RN]] = { norecurse nounwind readnone{{.*}} }
