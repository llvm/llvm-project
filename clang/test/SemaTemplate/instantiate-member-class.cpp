// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace PR8965 {
  template<typename T>
  struct X {
    typedef int type;

    T field; // expected-note{{in instantiation of member class}}
  };

  template<typename T>
  struct Y {
    struct Inner;

    typedef typename X<Inner>::type // expected-note{{in instantiation of template class}}
      type; // expected-note{{not-yet-instantiated member is declared here}}

    struct Inner {
      typedef type field; // expected-error{{no member 'type' in 'PR8965::Y<int>'; it has not yet been instantiated}}
    };
  };

  Y<int> y; // expected-note{{in instantiation of template class}}
}

template<typename T>
class X {
public:
  struct C { T &foo(); };

  struct D {
    struct E { T &bar(); }; // expected-error{{cannot form a reference to 'void'}}
    struct F; // expected-note{{member is declared here}}
  };
};

X<int>::C *c1;
X<float>::C *c2;

X<int>::X *xi; // expected-error{{qualified reference to 'X' is a constructor name rather than a type}}
X<float>::X *xf; // expected-error{{qualified reference to 'X' is a constructor name rather than a type}}

void test_naming() {
  c1 = c2; // expected-error{{assigning to 'X<int>::C *' from incompatible type 'X<float>::C *'}}
  xi = xf;  // expected-error{{assigning to 'X<int>::X<int> *' from incompatible type 'X<float>::X<float> *'}}
    // FIXME: error above doesn't print the type X<int>::X cleanly!
}

void test_instantiation(X<double>::C *x,
                        X<float>::D::E *e,
                        X<float>::D::F *f) {
  double &dr = x->foo();
  float &fr = e->bar();
  f->foo(); // expected-error{{implicit instantiation of undefined member 'X<float>::D::F'}}
  
}


X<void>::C *c3; // okay
X<void>::D::E *e1; // okay
X<void>::D::E e2; // expected-note{{in instantiation of member class 'X<void>::D::E' requested here}}

// Redeclarations.
namespace test1 {
  template <typename T> struct Registry {
    struct node;
    static node *Head;
    struct node {
      node(int v) { Head = this; }
    };
  };
  void test() {
    Registry<int>::node node(0);
  }
}

// Redeclarations during explicit instantiations.
namespace test2 {
  template <typename T> class A {
    class Foo;
    class Foo {
      int foo();
    };
  };
  template class A<int>;

  template <typename T> class B {
    class Foo;
    class Foo {
    public:
      typedef int X;
    };
    typename Foo::X x;
  };
  template class B<int>;

  template <typename T> class C {
    class Foo;
  };
  template <typename T> class C<T>::Foo {
    int x;
  };
  template class C<int>;
}

namespace AliasTagDef {
  template<typename T>
  struct F {
    using S = struct U {
#if __cplusplus <= 199711L
    // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif
      T g() {
        return T();
      }
    };
  };

  int m = F<int>::S().g();
  int n = F<int>::U().g();
}

namespace rdar10397846 {
  template<int I> struct A
  {
    struct B
    {
      struct C { C() { int *ptr = I; } };
#if __cplusplus >= 201103L
      // expected-error@-2 {{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
#else
      // expected-warning@-4 {{expression which evaluates to zero treated as a null pointer constant of type 'int *'}}
#endif
      // expected-error@-6 {{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
    };
  };

  template<int N> void foo()
  {
    class A<N>::B::C X; // expected-note 2 {{in instantiation of member function}}
    int A<N+1>::B::C::*member = 0;
  }

  void bar()
  {
    foo<0>(); // expected-note{{in instantiation of function template}}
    foo<1>(); // expected-note{{in instantiation of function template}}
  }
}
