// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic
template<typename T>
struct S {
 S() { }
};

template<typename T>
struct vector {
  void push_back(const T&) { int a[sizeof(T) ? -1: -1]; } // expected-error {{array with a negative size}}
};

class ExprEngine {
public:
 typedef vector<S<void *> >CheckersOrdered;
 CheckersOrdered Checkers;

 template <typename CHECKER>
 void registerCheck(CHECKER *check) {
   Checkers.push_back(S<void *>()); // expected-note {{in instantiation of member function 'vector<S<void *> >::push_back' requested here}}
 }
};

class RetainReleaseChecker { };

void f(ExprEngine& Eng) {
   Eng.registerCheck(new RetainReleaseChecker); // expected-note {{in instantiation of function template specialization 'ExprEngine::registerCheck<RetainReleaseChecker>' requested here}}
}

// PR 5838
namespace test1 {
  template<typename T> struct A {
    int a;
  };

  template<typename T> struct B : A<float>, A<T> {
    void f() {
      a = 0; // should not be ambiguous
    }
  };
  template struct B<int>;

  struct O {
    int a;
    template<typename T> struct B : A<T> {
      void f() {
        a = 0; // expected-error {{'test1::O::a' is not a member of class 'test1::O::B<int>'}}
      }
    };
  };
  template struct O::B<int>; // expected-note {{in instantiation}}
}

// PR7248
namespace test2 {
  template <class T> struct A {
    void foo() {
      T::bar(); // expected-error {{type 'int' cannot}}
    }
  };

  template <class T> class B {
    void foo(A<T> a) {
      a.test2::template A<T>::foo(); // expected-note {{in instantiation}}
    }
  };

  template class B<int>;
}

namespace PR14124 {
  template<typename T> struct S {
    int value;
  };
  template<typename T> void f() { S<T>::value; } // expected-error {{invalid use of non-static data member 'value'}}
  template void f<int>(); // expected-note {{in instantiation of}}

  struct List { List *next; };
  template<typename T, T *(T::*p) = &T::next> struct A {};
  A<List> a; // ok
  void operator&(struct Whatever);
  template<typename T, T *(T::*p) = &T::next> struct B {};
  B<List> b; // still ok
}
