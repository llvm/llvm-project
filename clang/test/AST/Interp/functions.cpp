// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -std=c++14 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s
// RUN: %clang_cc1 -std=c++14 -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

constexpr void doNothing() {}
constexpr int gimme5() {
  doNothing();
  return 5;
}
static_assert(gimme5() == 5, "");


template<typename T> constexpr T identity(T t) {
  static_assert(true, "");
  return t;
}
static_assert(identity(true), "");
static_assert(identity(true), ""); /// Compiled bytecode should be cached
static_assert(!identity(false), "");

template<typename A, typename B>
constexpr bool sameSize() {
  static_assert(sizeof(A) == sizeof(B), ""); // expected-error {{static assertion failed}} \
                                             // ref-error {{static assertion failed}} \
                                             // expected-note {{evaluates to}} \
                                             // ref-note {{evaluates to}}
  return true;
}
static_assert(sameSize<int, int>(), "");
static_assert(sameSize<unsigned int, int>(), "");
static_assert(sameSize<char, long>(), ""); // expected-note {{in instantiation of function template specialization}} \
                                           // ref-note {{in instantiation of function template specialization}}


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
  static_assert(foo() == 10, "");

  struct S {
    int i;
    void (*fp)();
  };

  constexpr S s{ 12 };
  static_assert(s.fp == nullptr, ""); // zero-initialized function pointer.

  constexpr int (*op)(int, int) = add;
  constexpr bool b = op;
  static_assert(op, "");
  static_assert(!!op, "");
  constexpr int (*op2)(int, int) = nullptr;
  static_assert(!op2, "");
}

namespace Comparison {
  void f(), g();
  constexpr void (*pf)() = &f, (*pg)() = &g;

  constexpr bool u13 = pf < pg; // ref-warning {{ordered comparison of function pointers}} \
                                // ref-error {{must be initialized by a constant expression}} \
                                // ref-note {{comparison between '&f' and '&g' has unspecified value}} \
                                // expected-warning {{ordered comparison of function pointers}} \
                                // expected-error {{must be initialized by a constant expression}} \
                                // expected-note {{comparison between '&f' and '&g' has unspecified value}}

  constexpr bool u14 = pf < (void(*)())nullptr; // ref-warning {{ordered comparison of function pointers}} \
                                                // ref-error {{must be initialized by a constant expression}} \
                                                // ref-note {{comparison between '&f' and 'nullptr' has unspecified value}} \
                                                // expected-warning {{ordered comparison of function pointers}} \
                                                // expected-error {{must be initialized by a constant expression}} \
                                                // expected-note {{comparison between '&f' and 'nullptr' has unspecified value}}



  static_assert(pf != pg, "");
  static_assert(pf == &f, "");
  static_assert(pg == &g, "");
}

}

struct F {
  constexpr bool ok() const {
    return okRecurse();
  }
  constexpr bool okRecurse() const {
    return true;
  }
};

struct BodylessMemberFunction {
  constexpr int first() const {
    return second();
  }
  constexpr int second() const {
    return 1;
  }
};

constexpr int nyd(int m);
constexpr int doit() { return nyd(10); }
constexpr int nyd(int m) { return m; }
static_assert(doit() == 10, "");

namespace InvalidCall {
  struct S {
    constexpr int a() const { // expected-error {{never produces a constant expression}} \
                              // ref-error {{never produces a constant expression}}
      return 1 / 0; // expected-note 2{{division by zero}} \
                    // expected-warning {{is undefined}} \
                    // ref-note 2{{division by zero}} \
                    // ref-warning {{is undefined}}
    }
  };
  constexpr S s;
  static_assert(s.a() == 1, ""); // expected-error {{not an integral constant expression}} \
                                 // expected-note {{in call to}} \
                                 // ref-error {{not an integral constant expression}} \
                                 // ref-note {{in call to}}

  /// This used to cause an assertion failure in the new constant interpreter.
  constexpr void func(); // expected-note {{declared here}} \
                         // ref-note {{declared here}}
  struct SS {
    constexpr SS() { func(); } // expected-note {{undefined function }} \
                               // ref-note {{undefined function}}
  };
  constexpr SS ss; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{in call to 'SS()'}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{in call to 'SS()'}}

}

namespace CallWithArgs {
  /// This used to call problems during checkPotentialConstantExpression() runs.
  constexpr void g(int a) {}
  constexpr void f() {
    g(0);
  }
}

namespace ReturnLocalPtr {
  constexpr int *p() {
    int a = 12;
    return &a; // ref-warning {{address of stack memory}} \
               // expected-warning {{address of stack memory}}
  }

  /// GCC rejects the expression below, just like the new interpreter. The current interpreter
  /// however accepts it and only warns about the function above returning an address to stack
  /// memory. If we change the condition to 'p() != nullptr', it even succeeds.
  static_assert(p() == nullptr, ""); // ref-error {{static assertion failed}} \
                                     // expected-error {{not an integral constant expression}}

  /// FIXME: The current interpreter emits diagnostics in the reference case below, but the
  /// new one does not.
  constexpr const int &p2() {
    int a = 12; // ref-note {{declared here}}
    return a; // ref-warning {{reference to stack memory associated with local variable}} \
              // expected-warning {{reference to stack memory associated with local variable}}
  }

  static_assert(p2() == 12, ""); // ref-error {{not an integral constant expression}} \
                                 // ref-note {{read of variable whose lifetime has ended}} \
                                 // expected-error {{not an integral constant expression}}
}

namespace VoidReturn {
  /// ReturnStmt with an expression in a void function used to cause problems.
  constexpr void bar() {}
  constexpr void foo() {
    return bar();
  }
  static_assert((foo(),1) == 1, "");
}

namespace InvalidReclRefs {
  void param(bool b) { // ref-note {{declared here}} \
                       // expected-note {{declared here}}
    static_assert(b, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{function parameter 'b' with unknown value}} \
                          // expected-error {{not an integral constant expression}} \
                          // expected-note {{function parameter 'b' with unknown value}}
    static_assert(true ? true : b, "");
  }

#if __cplusplus >= 202002L
  consteval void param2(bool b) { // ref-note {{declared here}} \
                                 // expected-note {{declared here}}
    static_assert(b, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{function parameter 'b' with unknown value}} \
                          // expected-error {{not an integral constant expression}} \
                          // expected-note {{function parameter 'b' with unknown value}}
  }
#endif
}

namespace TemplateUndefined {
  template<typename T> constexpr int consume(T);
  // ok, not a constant expression.
  const int k = consume(0);

  template<typename T> constexpr int consume(T) { return 0; }
  // ok, constant expression.
  constexpr int l = consume(0);
  static_assert(l == 0, "");
}
