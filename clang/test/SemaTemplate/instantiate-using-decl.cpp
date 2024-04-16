// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s

namespace test0 {
  namespace N { }

  template<typename T>
  struct A {
    void f();
  };

  template<typename T>
  struct B : A<T> {
    using A<T>::f;

    void g() {
      using namespace N;
      f();
    }
  };

  template struct B<int>;
}

namespace test1 {
  template <class Derived> struct Visitor1 {
    void Visit(struct Object1*);
  };
  template <class Derived> struct Visitor2 {
    void Visit(struct Object2*); // expected-note {{candidate function}}
  };

  template <class Derived> struct JoinVisitor
      : Visitor1<Derived>, Visitor2<Derived> {
    typedef Visitor1<Derived> Base1;
    typedef Visitor2<Derived> Base2;

    void Visit(struct Object1*);  // expected-note {{candidate function}}
    using Base2::Visit;
  };

  class Knot : public JoinVisitor<Knot> {
  };

  void test() {
    Knot().Visit((struct Object1*) 0);
    Knot().Visit((struct Object2*) 0);
    Knot().Visit((struct Object3*) 0); // expected-error {{no matching member function for call}}
  }
}

// PR5847
namespace test2 {
  namespace ns {
    void foo();
  }

  template <class T> void bar(T* ptr) {
    using ns::foo;
    foo();
  }

  template void bar(char *);
}

namespace test3 {
  template <typename T> struct t {
    struct s1 {
      T f1() const;
    };
    struct s2 : s1 {
      using s1::f1;
      T f1() const;
    };
  };

  void f2()
  {
    t<int>::s2 a;
    t<int>::s2 const & b = a;
    b.f1();
  }
}

namespace PR16936 {
  // Make sure both using decls are properly considered for
  // overload resolution.
  template<class> struct A {
    void access(int);
  };
  template<class> struct B {
    void access();
  };
  template<class CELL> struct X : public A<CELL>, public B<CELL> {
    using A<CELL>::access;
    using B<CELL>::access;

    void f() {
      access(0);
    }
  };

  void f() {
    X<int> x;
    x.f();
  }
}

namespace pr21923 {
template <typename> struct Base {
  int field;
  void method();
};
template <typename Scalar> struct Derived : Base<Scalar> {
  using Base<Scalar>::field;
  using Base<Scalar>::method;
  static void m_fn1() {
    // expected-error@+1 {{invalid use of member 'field' in static member function}}
    (void)field;
    // expected-error@+1 {{invalid use of member 'field' in static member function}}
    (void)&field;
    // expected-error@+1 {{call to non-static member function without an object argument}}
    (void)method;
    // expected-error@+1 {{call to non-static member function without an object argument}}
    (void)&method;
    // expected-error@+1 {{call to non-static member function without an object argument}}
    method();
    (void)&Base<Scalar>::field;
    (void)&Base<Scalar>::method;
  }
#if __cplusplus >= 201103L
  // These usages are OK in C++11 due to the unevaluated context.
  enum { TheSize = sizeof(field) };
  typedef decltype(field) U;
#else
  // expected-error@+1 {{invalid use of non-static data member 'field'}}
  enum { TheSize = sizeof(field) };
#endif
};

#if __cplusplus < 201103L
// C++98 has an extra note for TheSize.
// expected-note@+2 {{requested here}}
#endif
template class Derived<int>; // expected-note {{requested here}}

// This is interesting because we form an UnresolvedLookupExpr in the static
// function template and an UnresolvedMemberExpr in the instance function
// template. As a result, we get slightly different behavior.
struct UnresolvedTemplateNames {
  template <typename> void maybe_static();
#if __cplusplus < 201103L
  // expected-warning@+2 {{default template arguments for a function template are a C++11 extension}}
#endif
  template <typename T, typename T::type = 0> static void maybe_static();

  template <typename T>
  void instance_method() { (void)maybe_static<T>(); }
  template <typename T>
  static void static_method() {
    // expected-error@+1 {{call to non-static member function without an object argument}}
    (void)maybe_static<T>();
  }
};
void force_instantiation(UnresolvedTemplateNames x) {
  x.instance_method<int>();
  UnresolvedTemplateNames::static_method<int>(); // expected-note {{requested here}}
}
} // pr21923
