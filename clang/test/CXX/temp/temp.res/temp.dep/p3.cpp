// RUN: %clang_cc1 -fsyntax-only -verify %s
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

namespace PR14402 {
  template<typename T>
  struct A {
    typedef int n;
    int f();

    struct B {};
    struct C : B {
      // OK, can't be sure whether we derive from A yet.
      using A::n;
      int g() { return f(); }
    };

    struct D {
      using A::n; // expected-error {{using declaration refers into 'A<T>::', which is not a base class of 'D'}}
      int g() { return f(); } // expected-error {{call to non-static member function 'f' of 'A' from nested type 'D'}}
    };

    struct E { char &f(); };
    struct F : E {
      // FIXME: Reject this prior to instantiation; f() is known to return int.
      char &g() { return f(); }
      // expected-error@-1 {{'PR14402::A<int>::f' is not a member of class 'PR14402::A<int>::F'}}
      // expected-error@-2 {{non-const lvalue reference to type 'char' cannot bind to a temporary of type 'int'}}
    };
  };

  template<> struct A<int>::B : A<int> {};
  A<int>::C::n n = A<int>::C().g();

  // 'not a member'
  char &r = A<int>::F().g(); // expected-note {{in instantiation of}}
  template<> struct A<char>::E : A<char> {};
  // 'cannot bind to a temporary'
  char &s = A<char>::F().g(); // expected-note {{in instantiation of}}

  struct X;
  struct X { void f(); };
  struct X;
  template<typename T> struct Y : X {
    void g() {
      X::f();
    }
  };
}
