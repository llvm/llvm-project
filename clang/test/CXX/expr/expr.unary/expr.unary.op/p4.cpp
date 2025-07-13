// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  struct A {
    void foo(void (A::*)(int)); // expected-note {{passing argument to parameter here}}
    template<typename T> void g(T);

    void test() {
      foo(&g<int>); // expected-error-re {{cannot form member pointer of type 'void (A::*)(int){{( __attribute__\(\(thiscall\)\))?}}' without '&' and class name}}
    }
  };
}

// This should succeed.
namespace test1 {
  struct A {
    static void f(void (A::*)());
    static void f(void (*)(int));
    void g();
    static void g(int);

    void test() {
      f(&g);
    }
  };
}

namespace test2 {
  struct A {
    static int foo(short);
    static int foo(float);
    int foo(int);
    int foo(double);

    void test();
  };

  void A::test() {
    // FIXME: The error message in this case is less than clear, we can do
    // better.
    int (A::*ptr)(int) = &(A::foo); // expected-error {{cannot create a non-constant pointer to member function}}
  }
}

namespace GH40906 {
struct S {
    int x;
    void func();
    static_assert(__is_same_as(decltype((S::x)), int&), "");
    static_assert(__is_same_as(decltype(&(S::x)), int*), "");

    // FIXME: provide better error messages
    static_assert(__is_same_as(decltype((S::func)), int&), ""); // expected-error {{call to non-static member function without an object argument}}
    static_assert(__is_same_as(decltype(&(S::func)), int*), ""); // expected-error {{call to non-static member function without an object argument}}
};
static_assert(__is_same_as(decltype((S::x)), int&), "");
static_assert(__is_same_as(decltype(&(S::x)), int*), "");
static_assert(__is_same_as(decltype((S::func)), int&), ""); // expected-error {{call to non-static member function without an object argument}}
static_assert(__is_same_as(decltype(&(S::func)), int*), ""); // expected-error {{call to non-static member function without an object argument}}

struct A { int x;};

char q(int *);
short q(int A::*);

template <typename T>
constexpr int f(char (*)[sizeof(q(&T::x))]) { return 1; }

template <typename T>
constexpr int f(char (*)[sizeof(q(&(T::x)))]) { return 2; }

constexpr int g(char (*p)[sizeof(char)] = 0) { return f<A>(p); }
constexpr int h(char (*p)[sizeof(short)] = 0) { return f<A>(p); }

static_assert(g() == 2);
static_assert(h() == 1);

}
