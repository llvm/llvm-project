// RUN: %clang_cc1 %s -fsyntax-only -fms-extensions -std=c++11 -verify

struct Errors {
  using __super::foo; // expected-error {{'__super' cannot be used with a using declaration}}
  __super::XXX x; // expected-error {{invalid use of '__super', Errors has no base classes}} expected-error {{expected}}

  void foo() {
    // expected-note@+4 {{replace parentheses with an initializer to declare a variable}}
    // expected-warning@+3 {{empty parentheses interpreted as a function declaration}}
    // expected-error@+2 {{a type specifier is required for all declarations}}
    // expected-error@+1 {{use of '__super' inside a lambda is unsupported}}
    auto lambda = []{ __super::foo(); };
  }
};

struct Base1 {
  void foo(int) {}

  static void static_foo() {}

  typedef int XXX;
};

struct Derived : Base1 {
  __super::XXX x;
  typedef __super::XXX Type;

  enum E {
    X = sizeof(__super::XXX)
  };

  void foo() {
    __super::foo(1);

    if (true) {
      __super::foo(1);
    }

    return __super::foo(1);
  }

  static void bar() {
    __super::static_foo();
  }
};

struct Outer {
  struct Inner : Base1 {
    static const int x = sizeof(__super::XXX);
  };
};

struct Base2 {
  void foo(char) {}
};

struct MemberFunctionInMultipleBases : Base1, Base2 {
  void foo() {
    __super::foo('x');
  }
};

struct Base3 {
  void foo(int) {}
  void foo(char) {}
};

struct OverloadedMemberFunction : Base3 {
  void foo() {
    __super::foo('x');
  }
};

struct PointerToMember : Base1 {
  template <void (Base1::*MP)(int)>
  struct Wrapper {
    static void bar() {}
  };

  void baz();
};

void PointerToMember::baz() {
  Wrapper<&__super::foo>::bar();
}

template <typename T>
struct BaseTemplate {
  typedef int XXX;

  int foo() { return 0; }
};

struct DerivedFromKnownSpecialization : BaseTemplate<int> {
  __super::XXX a;
  typedef __super::XXX b;

  void foo() {
    __super::XXX c;
    typedef __super::XXX d;

    __super::foo();
  }
};

template <typename T>
struct DerivedFromDependentBase : BaseTemplate<T> {
  typename __super::XXX a;
  typedef typename __super::XXX b;

  __super::XXX c;         // expected-error {{missing 'typename'}}
  typedef __super::XXX d; // expected-error {{missing 'typename'}}

  void foo() {
    typename __super::XXX e;
    typedef typename __super::XXX f;

    __super::XXX g;         // expected-error {{missing 'typename'}}
    typedef __super::XXX h; // expected-error {{missing 'typename'}}

    int x = __super::foo();
  }
};

template <typename T>
struct DerivedFromTemplateParameter : T {
  typename __super::XXX a;
  typedef typename __super::XXX b;

  __super::XXX c;         // expected-error {{missing 'typename'}}
  typedef __super::XXX d; // expected-error {{missing 'typename'}}

  void foo() {
    typename __super::XXX e;
    typedef typename __super::XXX f;

    __super::XXX g;         // expected-error {{missing 'typename'}}
    typedef __super::XXX h; // expected-error {{missing 'typename'}}

    __super::foo(1);
  }
};

void instantiate() {
  DerivedFromDependentBase<int> d;
  d.foo();
  DerivedFromTemplateParameter<Base1> t;
  t.foo();
}

namespace {
struct B { int a; };
template <class C>
struct A : B {
  // Don't crash on dependent_type_var '->' '__super'
  void f() { int a = this->__super::a; }
};
}
