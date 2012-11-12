// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// pr6644

extern "C" {
  namespace N {
    struct X { 
      virtual void f();
    };
    void X::f() { }
  }
}

// CHECK: define void @_ZN1N1X1fEv
