// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
struct X0 {
  X0 f1();
  X0 f2();
};

template<typename T>
struct X1 {
  X1<T>(int);
  (X1<T>)(float);
  X1 f2();
  X1 f2(int);
  X1 f2(float);
};

// Error recovery: out-of-line constructors whose names have template arguments.
template<typename T> X1<T>::X1<T>(int) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}
template<typename T> (X1<T>::X1<T>)(float) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}

// Error recovery: out-of-line constructor names intended to be types
X0::X0 X0::f1() { return X0(); } // expected-error{{qualified reference to 'X0' is a constructor name rather than a type wherever a constructor can be declared}}

struct X0::X0 X0::f2() { return X0(); }

template<typename T> X1<T>::X1<T> X1<T>::f2() { } // expected-error{{qualified reference to 'X1' is a constructor name rather than a template name wherever a constructor can be declared}}
template<typename T> X1<T>::X1<T> (X1<T>::f2)(int) { } // expected-error{{qualified reference to 'X1' is a constructor name rather than a template name wherever a constructor can be declared}}
template<typename T> struct X1<T>::X1<T> (X1<T>::f2)(float) { }

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
    struct U : S { // expected-note 6{{candidate}}
      using S::S;
    };
    using T::T;
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
    using InhCtor::NS::NS; // expected-error {{using declaration in class refers into 'InhCtor::NS::', which is not a class}}

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
#endif
}
