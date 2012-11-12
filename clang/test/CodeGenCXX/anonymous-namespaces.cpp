// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -triple x86_64-apple-darwin10 -emit-llvm %s -o - > %t
// RUN: FileCheck %s -check-prefix=1 < %t
// RUN: FileCheck %s -check-prefix=2 < %t

int f();

namespace {
  // CHECK-1: @_ZN12_GLOBAL__N_11bE = internal global i32 0
  // CHECK-1: @_ZN12_GLOBAL__N_1L1cE = internal global i32 0
  // CHECK-1: @_ZN12_GLOBAL__N_11D1dE = internal global i32 0
  // CHECK-1: @_ZN12_GLOBAL__N_11aE = internal global i32 0
  int a = 0;

  int b = f();

  static int c = f();

  class D {
    static int d;
  };
  
  int D::d = f();

  // Check for generation of a VTT with internal linkage
  // CHECK-1: @_ZTSN12_GLOBAL__N_11X1EE = internal constant
  struct X { 
    struct EBase { };
    struct E : public virtual EBase { virtual ~E() {} };
  };

  // CHECK-1: define internal i32 @_ZN12_GLOBAL__N_13fooEv()
  int foo() {
    return 32;
  }

  // CHECK-1: define internal i32 @_ZN12_GLOBAL__N_11A3fooEv()
  namespace A {
    int foo() {
      return 45;
    }
  }
}

int concrete() {
  return a + foo() + A::foo();
}

void test_XE() { throw X::E(); }

// Miscompile on llvmc plugins.
namespace test2 {
  struct A {
    template <class T> struct B {
      static void foo() {}
    };
  };
  namespace {
    struct C;
  }

  // CHECK-2: define void @_ZN5test24testEv()
  // CHECK-2:   call void @_ZN5test21A1BINS_12_GLOBAL__N_11CEE3fooEv()
  void test() {
    A::B<C>::foo();
  }

  // CHECK-2: define internal void @_ZN5test21A1BINS_12_GLOBAL__N_11CEE3fooEv()
}
