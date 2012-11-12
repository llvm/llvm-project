// RUN: %clang_cc1 -fsyntax-only -verify %s 
typedef int INT;
typedef INT REALLY_INT; // expected-note {{previous definition is here}}
typedef REALLY_INT REALLY_REALLY_INT;
typedef REALLY_INT BOB;
typedef float REALLY_INT; // expected-error{{typedef redefinition with different types ('float' vs 'INT' (aka 'int'))}}

struct X {
  typedef int result_type; // expected-note {{previous definition is here}}
  typedef INT result_type; // expected-error {{redefinition of 'result_type'}}
};

struct Y; // expected-note{{previous definition is here}}
typedef int Y;  // expected-error{{typedef redefinition with different types ('int' vs 'Y')}}

typedef int Y2; // expected-note{{declared here}}
struct Y2; // expected-error{{definition of type 'Y2' conflicts with typedef of the same name}}

void f(); // expected-note{{previous definition is here}}
typedef int f; // expected-error{{redefinition of 'f' as different kind of symbol}}

typedef int f2; // expected-note{{previous definition is here}}
void f2(); // expected-error{{redefinition of 'f2' as different kind of symbol}}

typedef struct s s; 
typedef int I; 
typedef int I; 
typedef I I; 

struct s { };

// PR5874
namespace test1 {
  typedef int foo;
  namespace a { using test1::foo; };
  typedef int foo;
  using namespace a; 
  foo x;
}

namespace PR6923 {
  struct A;

  extern "C" {
    struct A;
    typedef struct A A;
  }

  struct A;
}

namespace PR7462 {
  struct A {};
  typedef int operator! (A); // expected-error{{typedef name must be an identifier}}
  int i = !A(); // expected-error{{invalid argument type}}
}

template<typename T>
typedef T f(T t) { return t; } // expected-error {{function definition declared 'typedef'}}
int k = f(0);
int k2 = k;

namespace PR11630 {
  template <class T>
  struct S
  {
    static const unsigned C = 1;
    static void f()
    {
      typedef int q[C == 1 ? 1 : -1]; // expected-note{{previous definition is here}}
      typedef int q[C >= 1 ? 2 : -2]; // expected-error{{typedef redefinition with different types ('int [2]' vs 'int [1]')}}
      typedef int n[C == 1 ? 1 : -1];
      typedef int n[C >= 1 ? 1 : -1];
    }
  };

  template <int T>
  struct S2
  {
    static void f()
    {
      typedef int q[1];  // expected-note{{previous definition is here}}
      typedef int q[T];  // expected-error{{typedef redefinition with different types ('int [2]' vs 'int [1]')}}
    }
  };

  void f() {
    S<int> a;
    a.f(); // expected-note{{in instantiation of member function 'PR11630::S<int>::f' requested here}}
    S2<1> b;
    b.f();
    S2<2> b2;
    b2.f(); // expected-note{{in instantiation of member function 'PR11630::S2<2>::f' requested here}}
  }
}
