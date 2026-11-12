// RUN: %clang_cc1 -fsyntax-only -Wpessimizing-move -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wpessimizing-move -std=c++11 -verify %s -DUSER_DEFINED
// RUN: %clang_cc1 -fsyntax-only -Wpessimizing-move -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
}
}

struct A {
#ifdef USER_DEFINED
  A() {}
  A(const A &) {}
  A(A &&) {}
  A &operator=(const A &) { return *this; }
  A &operator=(A &&) { return *this; }
#endif
};
struct B {
  B() {}
  B(A) {}
};

A test1(A a1) {
  A a2;
  return a1;
  return a2;
  return std::move(a1);
  return std::move(a2);
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:23}:""
}

B test2(A a1, B b1) {
  // Object is different than return type so don't warn.
  A a2;
  return a1;
  return a2;
  return std::move(a1);
  return std::move(a2);

  B b2;
  return b1;
  return b2;
  return std::move(b1);
  return std::move(b2);
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:23}:""

  return A();
  return test1(a2);
  return std::move(A());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  return std::move(test1(a2));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:29-[[@LINE-4]]:30}:""
}

A global_a;
A test3() {
  // Don't warn when object is not local.
  return global_a;
  return std::move(global_a);
  static A static_a;
  return static_a;
  return std::move(static_a);

}

A test4() {
  return A();
  return test3();

  return std::move(A());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  return std::move(test3());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:27-[[@LINE-4]]:28}:""
}

void test5(A) {
  test5(A());
  test5(test4());

  test5(std::move(A()));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:9-[[@LINE-3]]:19}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:23}:""
  test5(std::move(test4()));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:9-[[@LINE-3]]:19}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:26-[[@LINE-4]]:27}:""
}

void test6() {
  A a1 = A();
  A a2 = test3();

  A a3 = std::move(A());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""

  a3 = std::move(A());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:8-[[@LINE-3]]:18}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:22}:""

  A a4 = std::move(test3());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:27-[[@LINE-4]]:28}:""

  a4 = std::move(test3());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:8-[[@LINE-3]]:18}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:25-[[@LINE-4]]:26}:""
}

A test7() {
  A a1 = std::move(A());
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  A a2 = std::move((A()));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:25-[[@LINE-4]]:26}:""
  A a3 = (std::move(A()));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:21}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:24-[[@LINE-4]]:25}:""
  A a4 = (std::move((A())));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:21}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:26-[[@LINE-4]]:27}:""

  return std::move(a1);
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:23}:""
  return std::move((a1));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:24-[[@LINE-4]]:25}:""
  return (std::move(a1));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:21}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  return (std::move((a1)));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:21}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:25-[[@LINE-4]]:26}:""
}

#define wrap1(x) x
#define wrap2(x) x

// Macro test. Since the std::move call is outside the macro, it is
// safe to suggest a fix-it.
A test8() {
  A a;
  return std::move(a);
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:22}:""
  return std::move(wrap1(a));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:28-[[@LINE-4]]:29}:""
  return std::move(wrap1(wrap2(a)));
  // expected-warning@-1{{prevents copy elision}}
  // expected-note@-2{{remove std::move call}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:35-[[@LINE-4]]:36}:""
}

#define test9            \
  A test9() {            \
    A a;                 \
    return std::move(a); \
  }

// Macro test.  The std::call is inside the macro, so no fix-it is suggested.
test9
// expected-warning@-1{{prevents copy elision}}
// CHECK-NOT: fix-it

#define return_a return std::move(a)

// Macro test.  The std::call is inside the macro, so no fix-it is suggested.
A test10() {
  A a;
  return_a;
  // expected-warning@-1{{prevents copy elision}}
  // CHECK-NOT: fix-it
}

namespace templates {
  struct A {};
  struct B { B(A); };

  // Warn once here since the type is not dependent.
  template <typename T>
  A test1() {
    A a;
    return std::move(a);
    // expected-warning@-1{{prevents copy elision}}
    // expected-note@-2{{remove std::move call}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:22}:""
    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:24}:""
  }
  void run_test1() {
    test1<A>();
    test1<B>();
  }

  // T1 and T2 may not be the same, the warning may not always apply.
  template <typename T1, typename T2>
  T1 test2() {
    T2 t;
    return std::move(t);
  }
  void run_test2() {
    test2<A, A>();
    test2<B, A>();
  }
}
