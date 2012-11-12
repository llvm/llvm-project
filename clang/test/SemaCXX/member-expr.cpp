// RUN: %clang_cc1 -fsyntax-only -verify %s

class X{
public:
  enum E {Enumerator}; // expected-note 2{{declared here}}
  int f();
  static int mem;
  static float g();
};

void test(X* xp, X x) {
  int i1 = x.f();
  int i2 = xp->f();
  x.E; // expected-error{{cannot refer to type member 'E' in 'X' with '.'}}
  xp->E; // expected-error{{cannot refer to type member 'E' in 'X' with '->'}}
  int i3 = x.Enumerator;
  int i4 = xp->Enumerator;
  x.mem = 1;
  xp->mem = 2;
  float f1 = x.g();
  float f2 = xp->g();
}

struct A {
 int f0;
};
struct B {
 A *f0();
};
int f0(B *b) {
  return b->f0->f0; // expected-error{{did you mean to call it with no arguments}}
}

int i;

namespace C {
  int i;
}

void test2(X *xp) {
  xp->::i = 7; // expected-error{{qualified member access refers to a member in the global namespace}}
  xp->C::i = 7; // expected-error{{qualified member access refers to a member in namespace 'C'}}
}


namespace test3 {
  struct NamespaceDecl;

  struct NamedDecl {
    void *getIdentifier() const;
  };

  struct NamespaceDecl : NamedDecl {
    bool isAnonymousNamespace() const {
      return !getIdentifier();
    }
  };
}

namespace test4 {
  class X {
  protected:
    template<typename T> void f(T);
  };

  class Y : public X {
  public:
    using X::f;
  };

  void test_f(Y y) {
    y.f(17);
  }
}

namespace test5 {
  struct A {
    template <class T> void foo();
  };

  void test0(int x) {
    x.A::foo<int>(); // expected-error {{'int' is not a structure or union}}
  }

  void test1(A *x) {
    x.A::foo<int>(); // expected-error {{'test5::A *' is a pointer}}
  }

  void test2(A &x) {
    x->A::foo<int>(); // expected-error {{'test5::A' is not a pointer}}
  }
}

namespace PR7508 {
  struct A {
    struct CleanupScope {};
    void PopCleanupBlock(); // expected-note{{'PopCleanupBlock' declared here}}
  };

  void foo(A &a) {
    a.PopCleanupScope(); // expected-error{{no member named 'PopCleanupScope' in 'PR7508::A'; did you mean 'PopCleanupBlock'?}}
  }
}

namespace rdar8231724 {
  namespace N {
    template<typename T> struct X1;
    int i;
  }

  struct X { };
  struct Y : X { };

  void f(Y *y) {
    y->N::X1<int>; // expected-error{{'rdar8231724::N::X1' is not a member of class 'rdar8231724::Y'}}
  }
}

namespace PR9025 {
  struct S { int x; };
  S fun(); // expected-note{{possible target for call}}
  int fun(int i); // expected-note{{possible target for call}}
  int g() {
    return fun.x; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  }

  S fun2(); // expected-note{{possible target for call}}
  S fun2(int i); // expected-note{{possible target for call}}
  int g2() {
    return fun2.x; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  }

  S fun3(int i=0); // expected-note{{possible target for call}}
  int fun3(int i, int j); // expected-note{{possible target for call}}
  int g3() {
    return fun3.x; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  }

  template <typename T> S fun4(); // expected-note{{possible target for call}}
  int g4() {
    return fun4.x; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  }

  S fun5(int i); // expected-note{{possible target for call}}
  S fun5(float f); // expected-note{{possible target for call}}
  int g5() {
    return fun5.x; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  }
}

namespace FuncInMemberExpr {
  struct Vec { int size(); };
  Vec fun1();
  int test1() { return fun1.size(); } // expected-error {{base of member reference is a function; perhaps you meant to call it with no arguments}}
  Vec *fun2();
  int test2() { return fun2->size(); } // expected-error {{base of member reference is a function; perhaps you meant to call it with no arguments}}
  Vec fun3(int x = 0);
  int test3() { return fun3.size(); } // expected-error {{base of member reference is a function; perhaps you meant to call it with no arguments}}
}

namespace DotForSemiTypo {
void f(int i) {
  // If the programmer typo'd '.' for ';', make sure we point at the '.' rather
  // than the "field name" (whatever the first token on the next line happens to
  // be).
  int j = i. // expected-error {{member reference base type 'int' is not a structure or union}}
  j = 0;
}
}
