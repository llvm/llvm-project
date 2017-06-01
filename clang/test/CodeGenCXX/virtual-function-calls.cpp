// RUN: %clang_cc1 %s -triple %itanium_abi_triple -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -std=c++11 -emit-llvm -o - -fstrict-vtable-pointers -O1 | FileCheck --check-prefix=CHECK-INVARIANT %s

// PR5021
namespace PR5021 {

struct A {
  virtual void f(char);
};

void f(A *a) {
  // CHECK: call {{.*}}void %
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

namespace VirtualNoreturn {
  struct A {
    [[noreturn]] virtual void f();
  };

  // CHECK-LABEL: @_ZN15VirtualNoreturn1f
  // CHECK-INVARIANT-LABEL: define void @_ZN15VirtualNoreturn1f
  void f(A *p) {
    p->f();
    // CHECK: call {{.*}}void %{{[^#]*$}}
    // CHECK-NOT: unreachable
    // CHECK-INVARIANT: load {{.*}} !invariant.load ![[EMPTY_NODE:[0-9]+]]
  }
}

// CHECK-INVARIANT: ![[EMPTY_NODE]] = !{}
