// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

struct rdar9677163 {
  struct Y { ~Y(); };
  struct Z { ~Z(); };
  Y::~Y() { } // expected-error{{non-friend class member '~Y' cannot have a qualified name}}
  ~Z(); // expected-error{{expected the class name after '~' to name the enclosing class}}
};

namespace GH56772 {

template<class T>
struct A {
  ~A<T>();
};
#if __cplusplus >= 202002L
// FIXME: This isn't valid in C++20 and later.
#endif

struct B;

template<class T>
struct C {
  ~B(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};

template <typename T>
struct D {
  friend T::S::~S();
private:
  static constexpr int secret = 42;
};

template <typename T>
struct E {
  friend T::S::~V();
};

struct BadInstantiation {
  struct S {
    struct V {};
  };
};

struct GoodInstantiation {
  struct V {
    ~V();
  };
  using S = V;
};

// FIXME: We should diagnose this while instantiating.
E<BadInstantiation> x;
E<GoodInstantiation> y;

struct Q {
  struct S { ~S(); };
};

Q::S::~S() {
  void foo(int);
  foo(D<Q>::secret);
}

struct X {
  ~X();
};
struct Y;

struct Z1 {
  friend X::~Y(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};

template <class T>
struct Z2 {
  friend X::~Y(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};

}
