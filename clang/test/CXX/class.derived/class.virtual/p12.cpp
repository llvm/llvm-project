// RUN: %clang_cc1 -ast-print %s | FileCheck %s

// CHECK: test12_A::foo()
struct test12_A {
  virtual void foo();
  
  void bar() {
    test12_A::foo();
  }
};

// CHECK: xp->test24_B::wibble()
struct test24_B {
  virtual void wibble();
};

void foo(test24_B *xp) {
  xp->test24_B::wibble();
}
