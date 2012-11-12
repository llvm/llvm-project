// RUN: %clang_cc1 %s -emit-llvm-only -verify
// expected-no-diagnostics
// PR5489

template<typename E>
struct Bar {
 int x_;
};

static struct Bar<int> bar[1] = {
  { 0 }
};



namespace incomplete_type_refs {
  struct A;
  extern A g[];
  void foo(A*);
  void f(void) {
    foo(g);    // Reference to array with unknown element type.
  }

  struct A {   // define the element type.
    int a,b,c;
  };

  A *f2() {
    return &g[1];
  }

}

namespace PR10395 {
  struct T;
  extern T x[];
  T* f() { return x; }
}

namespace PR10384 {
  struct X;
  extern X x[1];
  X* f() { return x; }
}
