// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
struct A0 {
  struct K { };
};

template <typename T> struct B0: A0 {
  static void f() {
    K k;
  }
};

namespace E1 {
  typedef double A; 

  template<class T> class B {
    typedef int A; 
  };

  template<class T> 
  struct X : B<T> {
    A* blarg(double *dp) {
      return dp;
    }
  };
}

namespace E2 {
  struct A { 
    struct B;
    int *a;
    int Y;
  };
    
  int a;
  template<class T> struct Y : T { 
    struct B { /* ... */ };
    B b; 
    void f(int i) { a = i; } 
    Y* p;
  }; 
  
  Y<A> ya;
}
