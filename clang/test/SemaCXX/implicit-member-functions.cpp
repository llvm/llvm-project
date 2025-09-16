// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct A { };
A::A() { } // expected-error {{definition of implicitly declared default constructor}}

struct B { };
B::B(const B&) { } // expected-error {{definition of implicitly declared copy constructor}}

struct C { };
C& C::operator=(const C&) { return *this; } // expected-error {{definition of implicitly declared copy assignment operator}}

struct D { };
D::~D() { } // expected-error {{definition of implicitly declared destructor}}

// Make sure that the special member functions are introduced for
// name-lookup purposes and overload with user-declared
// constructors and assignment operators.
namespace PR6570 {
  class A { };

  class B {
  public:
    B() {}

    B(const A& a) {
      operator = (CONST);
      operator = (a);
    }

    B& operator = (const A& a) {
      return *this;
    }

    void f(const A &a) {
      B b(a);
    };

    static const B CONST;
  };

}

namespace PR7594 {
  // If the lazy declaration of special member functions is triggered
  // in an out-of-line initializer, make sure the functions aren't in
  // the initializer scope. This used to crash Clang:
  struct C {
    C();
    static C *c;
  };
  C *C::c = new C();
}

namespace Recursion {
  template<typename T> struct InvokeCopyConstructor {
    static const T &get();
    typedef decltype(T(get())) type;
  };
  struct B;
  struct A {
    typedef B type;
    template<typename T,
             typename = typename InvokeCopyConstructor<typename T::type>::type>
    A(const T &);
  };
  struct B {
    B();
    A a;
  };
  // Triggering the declaration of B's copy constructor causes overload
  // resolution to occur for A's copying constructor, which picks
  // the implicit copy constructor of A.
  // Because that copy constructor is always a perfect match the template
  // candidate is not instantiated.
  B b = B();


  // Another case, which isn't ill-formed under our rules. This is inspired by
  // a problem which occurs when combining CGAL with libstdc++-4.7.

  template<typename T> T &&declval();
  template<typename T, typename U> struct pair {
    pair();
    template<typename V, typename W,
             typename = decltype(T(declval<const V&>())),
             typename = decltype(U(declval<const W&>()))>
    pair(const pair<V,W> &);
  };

  template<typename K> struct Line;

  template<typename K> struct Vector {
    Vector(const Line<K> &l);
  };

  template<typename K> struct Point {
    Vector<K> v;
  };

  template<typename K> struct Line {
    pair<Point<K>, Vector<K>> x;
  };

  // Trigger declaration of Line copy ctor, which causes substitution into
  // pair's templated constructor, which triggers instantiation of the
  // definition of Point's copy constructor, which performs overload resolution
  // on Vector's constructors, which requires declaring all of Line's
  // constructors. That should not find a copy constructor (because we've not
  // declared it yet), but by the time we get all the way back here, we should
  // find the copy constructor.
  Line<void> L1;
  Line<void> L2(L1);
}
