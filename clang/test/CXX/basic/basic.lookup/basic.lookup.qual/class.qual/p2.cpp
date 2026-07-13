// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
struct X0 {
  X0();
  X0(int);
  X0 f1();
  X0 f2();
  typedef int A;
  typedef X0 B;
};

template<typename T>
struct X1 : X0 {
  X1();
  X1<T>(int);
  (X1<T>)(float);
  X1 f2();
  X1 f2(int);
  X1 f2(float);
  X1 f2(double);
  X1 f2(short);
  X1 f2(long);
};

// Error recovery: out-of-line constructors whose names have template arguments.
template<typename T> X1<T>::X1<T>(int) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}
template<typename T> (X1<T>::X1<T>)(float) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}

// Error recovery: out-of-line constructor names intended to be types
X0::X0 X0::f1() { return X0(); } // expected-error{{qualified reference to 'X0' is a constructor name rather than a type in this context}}

struct X0::X0 X0::f2() { return X0(); }

template<typename T> X1<T>::X1<T> X1<T>::f2() { } // expected-warning{{missing 'typename'}}
template<typename T> X1<T>::X1<T> (X1<T>::f2)(int) { } // expected-warning{{missing 'typename'}}
template<typename T> struct X1<T>::X1<T> (X1<T>::f2)(float) { }
template<typename T> struct X1<T>::X1 (X1<T>::f2)(double) { }
template<typename T> typename X1<T>::template X1<T> X1<T>::f2(short) { } // expected-warning {{qualified reference to 'X1' is a constructor name rather than a template name in this context}}
template<typename T> typename X1<T>::template X1<T> (X1<T>::f2)(long) { } // expected-warning {{qualified reference to 'X1' is a constructor name rather than a template name in this context}}

void x1test(X1<int> x1i) {
  x1i.f2();
  x1i.f2(0);
  x1i.f2(0.f);
  x1i.f2(0.);
}

void other_contexts() {
  X0::X0 x0; // expected-error{{qualified reference to 'X0' is a constructor name rather than a type in this context}}
  X1<int>::X1 x1a; // expected-error{{qualified reference to 'X1' is a constructor name rather than a type in this context}}
  X1<int>::X1<float> x1b; // expected-error{{qualified reference to 'X1' is a constructor name rather than a template name in this context}}

  X0::B ok1;
  X0::X0::A ok2;
  X0::X0::X0 x0b; // expected-error{{qualified reference to 'X0' is a constructor name rather than a type in this context}}
  X1<int>::X0 ok3;
  X1<int>::X0::X0 x0c; // expected-error{{qualified reference to 'X0' is a constructor name rather than a type in this context}}
  X1<int>::X1<float>::X0 ok4;

  {
    typename X0::X0 tn1; // expected-warning{{qualified reference to 'X0' is a constructor name rather than a type in this context}} expected-warning 0-1{{typename}}
    typename X1<int>::X1<float> tn2; // expected-warning{{qualified reference to 'X1' is a constructor name rather than a template name in this context}} expected-warning 0-1{{typename}}
    typename X0::B ok1; // expected-warning 0-1{{typename}}
    typename X1<int>::X0 ok2; // expected-warning 0-1{{typename}}
  }

  {
    struct X0::X0 tag1;
    struct X1<int>::X1 tag2;
    struct X1<int>::X1<int> tag3;
  }

  int a;
  {
    X0::X0(a); // expected-error{{qualified reference to 'X0' is a constructor name rather than a type in this context}}
    // expected-warning@-1 {{redundant parentheses around declaration of variable named 'a'}} expected-note@-1 2{{}}
  }
}

template<typename T> void in_instantiation_x0() {
  typename T::X0 x0; // expected-warning{{qualified reference to 'X0' is a constructor name rather than a type in this context}}
  typename T::A a;
  typename T::B b;
}
template void in_instantiation_x0<X0>(); // expected-note {{instantiation of}}

template<typename T> void in_instantiation_x1() {
  typename T::X1 x1; // expected-warning{{qualified reference to 'X1' is a constructor name rather than a type in this context}}
  typename T::template X1<int> x1i; // expected-warning{{qualified reference to 'X1' is a constructor name rather than a template name in this context}}
  typename T::X0 x0;
}
template void in_instantiation_x1<X1<int> >(); // expected-note {{instantiation of}}

namespace sfinae {
  template<typename T> void f(typename T::X0 *) = delete; // expected-warning 0-1{{extension}}
  template<typename T> void f(...);
  void g() { f<X0>(0); }
}

namespace versus_injected_class_name {
  template <typename T> struct A : T::B {
    struct T::B *p;
    typename T::B::type a;
    A() : T::B() {}

    typename T::B b; // expected-warning {{qualified reference to 'B' is a constructor name rather than a type in this context}}
  };
  struct B {
    typedef int type;
  };
  template struct A<B>; // expected-note {{in instantiation of}}
}

// We have a special case for lookup within using-declarations that are
// member-declarations: foo::bar::baz::baz always names baz's constructor
// in such a context, even if looking up 'baz' within foo::bar::baz would
// not find the injected-class-name. Likewise foo::bar::baz<T>::baz also
// names the constructor.
namespace InhCtor {
  struct A {
    A(int);
  protected:
    int T();
  };
  typedef A T;
  struct B : A {
    // This is a using-declaration for 'int A::T()' in C++98, but is an
    // inheriting constructor declaration in C++11.
    using InhCtor::T::T;
  };
#if __cplusplus < 201103L
  B b(123);      // expected-error {{no matching constructor}}
                 // expected-note@-7 2{{candidate constructor}}
  int n = b.T(); // ok, accessible
#else
  B b(123);      // ok, inheriting constructor
  int n = b.T(); // expected-error {{'T' is a protected member of 'InhCtor::A'}}
                 // expected-note@-15 {{declared protected here}}

  // FIXME: EDG and GCC reject this too, but it's not clear why it would be
  // ill-formed.
  template<typename T>
  struct S : T {
    struct U; // expected-note 6{{candidate}}
    using T::T;
  };

  template<typename T>
  struct S<T>::U : S {
    using S::S;
  };

  S<A>::U ua(0); // expected-error {{no match}}
  S<B>::U ub(0); // expected-error {{no match}}

  template<typename T>
  struct X : T {
    using T::Z::U::U;
  };
  template<typename T>
  struct X2 : T {
    using T::Z::template V<int>::V;
  };
  struct Y {
    struct Z {
      typedef Y U;
      template<typename T> using V = Y;
    };
    Y(int);
  };
  X<Y> xy(0);

  namespace Repeat {
    struct A {
      struct T {
        T(int);
      };
    };
    struct Z : A {
      using A::A::A;
    };
    template<typename T>
    struct ZT : T::T {
      using T::T::T;
    };
  }

  namespace NS {
    struct NS {};
  }
  struct DerivedFromNS : NS::NS {
    // No special case unless the NNS names a class.
    using InhCtor::NS::NS; // expected-error {{using declaration in class refers into 'InhCtor::NS', which is not a class}}

  };

  // FIXME: Consider reusing the same diagnostic between dependent and non-dependent contexts
  typedef int I;
  struct UsingInt {
    using I::I; // expected-error {{'I' (aka 'int') is not a class, namespace, or enumeration}}
  };
  template<typename T> struct UsingIntTemplate {
    using T::T; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
  };
  UsingIntTemplate<int> uit; // expected-note {{here}}

  // This case is odd: we don't name the constructor of a dependent base as
  // Base::Base, but we still happen to have enough information to identify
  // when parsing the template that we're inheriting constructors.
  //
  // FIXME: Once CWG 2070 is resolved, check whether this case should be
  // accepted or not.
  namespace DependentCtorName {
    template <typename T> struct B { B(int); };
    template <typename T> struct A : B<T> {
      using X = B<T>;
      using X::B;
    };
    A<int> ab = 0;
  }
#endif
}
