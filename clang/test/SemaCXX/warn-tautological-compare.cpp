// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -verify -std=c++11 %s

namespace RuntimeBehavior {
  // Avoid emitting tautological compare warnings when the code already has
  // compile time checks on variable sizes.

  const int kintmax = 2147483647;
  void test0(short x) {
    if (sizeof(x) < sizeof(int) || x < kintmax) {}

    if (x < kintmax) {}
    // expected-warning@-1{{comparison of constant 2147483647 with expression of type 'short' is always true}}
  }

  void test1(short x) {
    if (x < kintmax) {}
    // expected-warning@-1{{comparison of constant 2147483647 with expression of type 'short' is always true}}

    if (sizeof(x) < sizeof(int))
      return;

    if (x < kintmax) {}
  }
}

namespace ArrayCompare {
  #define GetValue(ptr)  ((ptr != 0) ? ptr[0] : 0)
  extern int a[] __attribute__((weak));
  int b[] = {8,13,21};
  struct {
    int x[10];
  } c;
  const char str[] = "text";
  void ignore() {
    if (a == 0) {}
    if (a != 0) {}
    (void)GetValue(b);
  }
  void test() {
    if (b == 0) {}
    // expected-warning@-1{{comparison of array 'b' equal to a null pointer is always false}}
    if (b != 0) {}
    // expected-warning@-1{{comparison of array 'b' not equal to a null pointer is always true}}
    if (0 == b) {}
    // expected-warning@-1{{comparison of array 'b' equal to a null pointer is always false}}
    if (0 != b) {}
    // expected-warning@-1{{comparison of array 'b' not equal to a null pointer is always true}}
    if (c.x == 0) {}
    // expected-warning@-1{{comparison of array 'c.x' equal to a null pointer is always false}}
    if (c.x != 0) {}
    // expected-warning@-1{{comparison of array 'c.x' not equal to a null pointer is always true}}
    if (str == 0) {}
    // expected-warning@-1{{comparison of array 'str' equal to a null pointer is always false}}
    if (str != 0) {}
    // expected-warning@-1{{comparison of array 'str' not equal to a null pointer is always true}}
  }
}

namespace FunctionCompare {
  #define CallFunction(f) ((f != 0) ? f() : 0)
  extern void a()  __attribute__((weak));
  void fun1();
  int fun2();
  int* fun3();
  int* fun4(int);
  class S {
  public:
    static int foo();
  };
  void ignore() {
    if (a == 0) {}
    if (0 != a) {}
    (void)CallFunction(fun2);
  }
  void test() {
    if (fun1 == 0) {}
    // expected-warning@-1{{comparison of function 'fun1' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    if (fun2 == 0) {}
    // expected-warning@-1{{comparison of function 'fun2' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    // expected-note@-3{{suffix with parentheses to turn this into a function call}}
    if (fun3 == 0) {}
    // expected-warning@-1{{comparison of function 'fun3' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    // expected-note@-3{{suffix with parentheses to turn this into a function call}}
    if (fun4 == 0) {}
    // expected-warning@-1{{comparison of function 'fun4' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    if (nullptr != fun1) {}
    // expected-warning@-1{{comparison of function 'fun1' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    if (nullptr != fun2) {}
    // expected-warning@-1{{comparison of function 'fun2' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    if (nullptr != fun3) {}
    // expected-warning@-1{{comparison of function 'fun3' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    // expected-note@-3{{suffix with parentheses to turn this into a function call}}
    if (nullptr != fun4) {}
    // expected-warning@-1{{comparison of function 'fun4' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    if (S::foo == 0) {}
    // expected-warning@-1{{comparison of function 'S::foo' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    // expected-note@-3{{suffix with parentheses to turn this into a function call}}
  }
}

namespace PointerCompare {
  extern int a __attribute__((weak));
  int b;
  static int c;
  class S {
  public:
    static int a;
    int b;
  };
  void ignored() {
    if (&a == 0) {}
  }
  void test() {
    S s;
    if (&b == 0) {}
    // expected-warning@-1{{comparison of address of 'b' equal to a null pointer is always false}}
    if (&c == 0) {}
    // expected-warning@-1{{comparison of address of 'c' equal to a null pointer is always false}}
    if (&s.a == 0) {}
    // expected-warning@-1{{comparison of address of 's.a' equal to a null pointer is always false}}
    if (&s.b == 0) {}
    // expected-warning@-1{{comparison of address of 's.b' equal to a null pointer is always false}}
    if (&S::a == 0) {}
    // expected-warning@-1{{comparison of address of 'S::a' equal to a null pointer is always false}}
  }
}

namespace macros {
  #define assert(x) if (x) {}
  int array[5];
  void fun();
  int x;

  void test() {
    assert(array == 0);
    // expected-warning@-1{{comparison of array 'array' equal to a null pointer is always false}}
    assert(array != 0);
    // expected-warning@-1{{comparison of array 'array' not equal to a null pointer is always true}}
    assert(array == 0 && "expecting null pointer");
    // expected-warning@-1{{comparison of array 'array' equal to a null pointer is always false}}
    assert(array != 0 && "expecting non-null pointer");
    // expected-warning@-1{{comparison of array 'array' not equal to a null pointer is always true}}

    assert(fun == 0);
    // expected-warning@-1{{comparison of function 'fun' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    assert(fun != 0);
    // expected-warning@-1{{comparison of function 'fun' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    assert(fun == 0 && "expecting null pointer");
    // expected-warning@-1{{comparison of function 'fun' equal to a null pointer is always false}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}
    assert(fun != 0 && "expecting non-null pointer");
    // expected-warning@-1{{comparison of function 'fun' not equal to a null pointer is always true}}
    // expected-note@-2{{prefix with the address-of operator to silence this warning}}

    assert(&x == 0);
    // expected-warning@-1{{comparison of address of 'x' equal to a null pointer is always false}}
    assert(&x != 0);
    // expected-warning@-1{{comparison of address of 'x' not equal to a null pointer is always true}}
    assert(&x == 0 && "expecting null pointer");
    // expected-warning@-1{{comparison of address of 'x' equal to a null pointer is always false}}
    assert(&x != 0 && "expecting non-null pointer");
    // expected-warning@-1{{comparison of address of 'x' not equal to a null pointer is always true}}
  }
}
