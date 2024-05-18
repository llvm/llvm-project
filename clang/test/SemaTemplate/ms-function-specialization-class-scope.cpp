// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -fms-extensions -fdelayed-template-parsing -fsyntax-only -verify %s

// expected-no-diagnostics
class A {
public:
  template<class U> A(U p) {}
  template<> A(int p) {}

  template<class U> void f(U p) {}

  template<> void f(int p) {}

  void f(int p) {}
};

void test1() {
  A a(3);
  char *b;
  a.f(b);
  a.f<int>(99);
  a.f(100);
}

template<class T> class B {
public:
  template<class U> B(U p) {}
  template<> B(int p) {}

  template<class U> void f(U p) { T y = 9; }

  template<> void f(int p) {
    T a = 3;
  }

  void f(int p) { T a = 3; }
};

void test2() {
  B<char> b(3);
  char *ptr;
  b.f(ptr);
  b.f<int>(99);
  b.f(100);
}

namespace PR12709 {
  template<class T> class TemplateClass {
    void member_function() { specialized_member_template<false>(); }

    template<bool b> void specialized_member_template() {}

    template<> void specialized_member_template<false>() {}
  };

  void f() { TemplateClass<int> t; }
}

namespace Duplicates {
  template<typename T> struct A {
    template<typename U> void f();
    template<> void f<int>() {}
    template<> void f<T>() {}
  };

  // FIXME: We should diagnose the duplicate explicit specialization definitions
  // here.
  template struct A<int>;
}

namespace PR28082 {
struct S {
  template <int>
  int f(int = 0);
  template <>
  int f<0>(int);
};
}
