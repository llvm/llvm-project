// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

constexpr void doNothing() {}
constexpr int gimme5() {
  doNothing();
  return 5;
}
static_assert(gimme5() == 5, "");


template<typename T> constexpr T identity(T t) { return t; }
static_assert(identity(true), "");
static_assert(identity(true), ""); /// Compiled bytecode should be cached
static_assert(!identity(false), "");

constexpr auto add(int a, int b) -> int {
  return identity(a) + identity(b);
}

constexpr int sub(int a, int b) {
  return a - b;
}
static_assert(sub(5, 2) == 3, "");
static_assert(sub(0, 5) == -5, "");

constexpr int norm(int n) {
  if (n >= 0) {
    return identity(n);
  }
  return -identity(n);
}
static_assert(norm(5) == norm(-5), "");

constexpr int square(int n) {
  return norm(n) * norm(n);
}
static_assert(square(2) == 4, "");

constexpr int add_second(int a, int b, bool doAdd = true) {
  if (doAdd)
    return a + b;
  return a;
}
static_assert(add_second(10, 3, true) == 13, "");
static_assert(add_second(10, 3) == 13, "");
static_assert(add_second(300, -20, false) == 300, "");


constexpr int sub(int a, int b, int c) {
  return a - b - c;
}
static_assert(sub(10, 8, 2) == 0, "");


constexpr int recursion(int i) {
  doNothing();
  i = i - 1;
  if (i == 0)
    return identity(0);

  return recursion(i);
}
static_assert(recursion(10) == 0, "");

template<int N = 5>
constexpr decltype(N) getNum() {
  return N;
}
static_assert(getNum<-2>() == -2, "");
static_assert(getNum<10>() == 10, "");
static_assert(getNum() == 5, "");

constexpr int f(); // expected-note {{declared here}} \
                   // ref-note {{declared here}}
static_assert(f() == 5, ""); // expected-error {{not an integral constant expression}} \
                             // expected-note {{undefined function 'f'}} \
                             // ref-error {{not an integral constant expression}} \
                             // ref-note {{undefined function 'f'}}
constexpr int a() {
  return f();
}
constexpr int f() {
  return 5;
}
static_assert(a() == 5, "");

constexpr int invalid() {
  // Invalid expression in visit().
  while(huh) {} // expected-error {{use of undeclared identifier}} \
                // ref-error {{use of undeclared identifier}}

  return 0;
}

constexpr void invalid2() {
  int i = 0;
  // Invalid expression in discard().
  huh(); // expected-error {{use of undeclared identifier}} \
         // ref-error {{use of undeclared identifier}}
}

namespace FunctionPointers {
  constexpr int add(int a, int b) {
    return a + b;
  }

  struct S { int a; };
  constexpr S getS() {
    return S{12};
  }

  constexpr int applyBinOp(int a, int b, int (*op)(int, int)) {
    return op(a, b);
  }
  static_assert(applyBinOp(1, 2, add) == 3, "");

  constexpr int ignoreReturnValue() {
    int (*foo)(int, int) = add;

    foo(1, 2);
    return 1;
  }
  static_assert(ignoreReturnValue() == 1, "");

  constexpr int createS(S (*gimme)()) {
    gimme(); // Ignored return value
    return gimme().a;
  }
  static_assert(createS(getS) == 12, "");

namespace FunctionReturnType {
  typedef int (*ptr)(int*);
  typedef ptr (*pm)();

  constexpr int fun1(int* y) {
      return *y + 10;
  }
  constexpr ptr fun() {
      return &fun1;
  }
  static_assert(fun() == nullptr, ""); // expected-error {{static assertion failed}} \
                                       // ref-error {{static assertion failed}}

  constexpr int foo() {
    int (*f)(int *) = fun();
    int m = 0;

    m = f(&m);

    return m;
  }
  static_assert(foo() == 10);

  struct S {
    int i;
    void (*fp)();
  };

  constexpr S s{ 12 };
  static_assert(s.fp == nullptr); // zero-initialized function pointer.
}

}
