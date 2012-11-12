// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// PR5021
namespace PR5021 {

struct A {
  virtual void f(char);
};

void f(A *a) {
  // CHECK: call void %
  a->f('c');
}

struct B : virtual A { 
  virtual void f();
};

void f(B * b) {
  b->f();
}

}

namespace Test1 {
  struct A { 
    virtual ~A(); 
  };

  struct B : A {
    virtual ~B();
    virtual void f();
  };

  void f(B *b) {
    b->f();
  }
}
