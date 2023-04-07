// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct A {
  struct B { };
  
  friend struct B;
};

void f() {
  A<int>::B b;
}

struct C0 {
  friend struct A<int>;
};

namespace PR6770 {
  namespace N {
    int f1(int);
  }
  using namespace N;

  namespace M { 
    float f1(float);
  }
  using M::f1;

  template<typename T> void f1(T, T);
  template <class T>
  void f() {
    friend class f; // expected-error{{'friend' used outside of class}}
    friend class f1; // expected-error{{'friend' used outside of class}}
  }
}

namespace friend_redecl_inline {
// We had a bug where instantiating the foo friend declaration would check the
// defined-ness of the most recent decl while checking if the canonical decl was
// inlined.
void foo();
void bar();
template <typename T>
class C {
  friend void foo();
  friend inline void bar();
};
inline void foo() {}
inline void bar() {}
C<int> c;
}

namespace qualified_friend {
  void f(int); // expected-note 2{{type mismatch at 1st parameter}}
  template<typename T> void f(T*); // expected-note 2{{could not match 'T *' against 'double'}}
  template<typename T> void nondep();

  template<typename> struct X1 {
    friend void qualified_friend::f(double); // expected-error {{friend declaration of 'f' does not match any declaration in namespace 'qualified_friend'}}
    friend void qualified_friend::g(); // expected-error {{friend declaration of 'g' does not match any declaration in namespace 'qualified_friend'}}
  };
  template<typename T> struct X2 {
    friend void qualified_friend::f(T); // expected-error {{friend declaration of 'f' does not match any declaration in namespace 'qualified_friend'}}
  };
  X1<int> xi;
  X2<double> xd; // expected-note {{in instantiation of}}
  X2<int> x2i;

  struct Y {
    void f(int); // expected-note 2{{type mismatch at 1st parameter}}
    template<typename T> void f(T*); // expected-note 2{{could not match 'T *' against 'double'}}
    template<typename T> void nondep();
  };

  template<typename> struct Z1 {
    friend void Y::f(double); // expected-error {{friend declaration of 'f' does not match any declaration in 'qualified_friend::Y'}}
    friend void Y::g(); // expected-error {{friend declaration of 'g' does not match any declaration in 'qualified_friend::Y'}}
  };
  template<typename T> struct Z2 {
    friend void Y::f(T); // expected-error {{friend declaration of 'f' does not match any declaration in 'qualified_friend::Y'}}
  };
  Z1<int> zi;
  Z2<double> zd; // expected-note {{in instantiation of}}
  Z2<int> z2i;

  template<typename T>
  struct OK {
    friend void qualified_friend::f(int);
    friend void qualified_friend::f(int*);
    friend void qualified_friend::f(T*);
    friend void qualified_friend::f<T>(T*);
    friend void qualified_friend::nondep<int>();
    friend void qualified_friend::nondep<T>();

    friend void Y::f(int);
    friend void Y::f(int*);
    friend void Y::f(T*);
    friend void Y::f<T>(T*);
    friend void Y::nondep<int>();
    friend void Y::nondep<T>();
  };
  OK<float> ok;
}

namespace qualified_friend_finds_nothing {
  // FIXME: The status of this example is unclear. For now, we diagnose if the
  // qualified declaration has nothing it can redeclare, but allow qualified
  // lookup to find later-declared function templates during instantiation.
  //
  // This matches the behavior of GCC, EDG, ICC, and MSVC (except that GCC and
  // ICC bizarrely accept the instantiation of B<float>).
  namespace N {}

  template<typename T> struct A {
    friend void N::f(T); // expected-error {{friend declaration of 'f' does not match}}
  };
  namespace N { void f(); } // expected-note {{different number of parameters}}

  template<typename T> struct B {
    friend void N::f(T); // expected-error {{friend declaration of 'f' does not match}}
  };
  B<float> bf; // expected-note {{in instantiation of}}

  namespace N { void f(int); }
  B<int> bi; // ok?!
}

namespace PR37556 {
  inline namespace N { int x1, x2, y1, y2; } // expected-note 2{{previous}}
  struct X {
    friend void x1(int);
    friend void PR37556::x2(int); // expected-error {{different kind}}
  };
  template<typename T> struct Y {
    friend void y1(T);
    friend void PR37556::y2(T); // expected-error {{different kind}}
  };
  template struct Y<int>;
  template<typename T> struct Z {
    friend void z1(T);
    friend void PR37556::z2(T); // expected-error {{does not match any}}
  };
  inline namespace N { int z1, z2; }
  template struct Z<int>;
}

namespace PR42513_comment3 {
  template<typename X> struct T { friend auto f(X*) { return nullptr; } };
  struct X1 { friend auto f(X1*); };
  template struct T<X1>;
  int n = f((X1*)nullptr); // expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'std::nullptr_t'}}
}

namespace GH61851 {
namespace A {
inline namespace B {
  inline constexpr struct {} foo;
}

template <typename T>
class Bar {
  template <typename U>
  friend void foo(U &&arg) {} // no diagnostic expected
};
}

void foobar() {
  A::Bar<int> b;
}
}
