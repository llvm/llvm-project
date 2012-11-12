// RUN: %clang_cc1 -fsyntax-only -verify %s

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
