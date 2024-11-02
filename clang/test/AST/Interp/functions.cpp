// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++14 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s
// RUN: %clang_cc1 -std=c++14 -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both %s

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
  static_assert(sizeof(A) == sizeof(B), ""); // both-error {{static assertion failed}} \
                                             // both-note {{evaluates to}}
  return true;
}
static_assert(sameSize<int, int>(), "");
static_assert(sameSize<unsigned int, int>(), "");
static_assert(sameSize<char, long>(), ""); // both-note {{in instantiation of function template specialization}}


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

constexpr int f(); // both-note {{declared here}}
static_assert(f() == 5, ""); // both-error {{not an integral constant expression}} \
                             // both-note {{undefined function 'f'}}
constexpr int a() {
  return f();
}
constexpr int f() {
  return 5;
}
static_assert(a() == 5, "");

constexpr int invalid() {
  // Invalid expression in visit().
  while(huh) {} // both-error {{use of undeclared identifier}}
  return 0;
}

constexpr void invalid2() {
  int i = 0;
  // Invalid expression in discard().
  huh(); // both-error {{use of undeclared identifier}}
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
    return op(a, b); // both-note {{evaluates to a null function pointer}}
  }
  static_assert(applyBinOp(1, 2, add) == 3, "");
  static_assert(applyBinOp(1, 2, nullptr) == 3, ""); // both-error {{not an integral constant expression}} \
                                                     // both-note {{in call to}}


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
  static_assert(fun() == nullptr, ""); // both-error {{static assertion failed}}

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

  int m() { return 5;} // both-note {{declared here}}
  constexpr int (*invalidFnPtr)() = m;
  static_assert(invalidFnPtr() == 5, ""); // both-error {{not an integral constant expression}} \
                                          // both-note {{non-constexpr function 'm'}}
}

namespace Comparison {
  void f(), g();
  constexpr void (*pf)() = &f, (*pg)() = &g;

  constexpr bool u13 = pf < pg; // both-warning {{ordered comparison of function pointers}} \
                                // both-error {{must be initialized by a constant expression}} \
                                // both-note {{comparison between '&f' and '&g' has unspecified value}}

  constexpr bool u14 = pf < (void(*)())nullptr; // both-warning {{ordered comparison of function pointers}} \
                                                // both-error {{must be initialized by a constant expression}} \
                                                // both-note {{comparison between '&f' and 'nullptr' has unspecified value}}



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
    constexpr int a() const { // both-error {{never produces a constant expression}}
      return 1 / 0; // both-note 2{{division by zero}} \
                    // both-warning {{is undefined}}
    }
  };
  constexpr S s;
  static_assert(s.a() == 1, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{in call to}}

  /// This used to cause an assertion failure in the new constant interpreter.
  constexpr void func(); // both-note {{declared here}}
  struct SS {
    constexpr SS() { func(); } // both-note {{undefined function }}
  };
  constexpr SS ss; // both-error {{must be initialized by a constant expression}} \
                   // both-note {{in call to 'SS()'}}


  /// This should not emit a diagnostic.
  constexpr int f();
  constexpr int a() {
    return f();
  }
  constexpr int f() {
    return 5;
  }
  static_assert(a() == 5, "");

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
    return &a; // both-warning {{address of stack memory}}
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
    return a; // both-warning {{reference to stack memory associated with local variable}}
  }

  static_assert(p2() == 12, ""); // both-error {{not an integral constant expression}} \
                                 // ref-note {{read of variable whose lifetime has ended}}
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
  void param(bool b) { // both-note {{declared here}}
    static_assert(b, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{function parameter 'b' with unknown value}}
    static_assert(true ? true : b, "");
  }

#if __cplusplus >= 202002L
  consteval void param2(bool b) { // both-note {{declared here}}
    static_assert(b, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{function parameter 'b' with unknown value}}
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

namespace PtrReturn {
  constexpr void *a() {
    return nullptr;
  }
  static_assert(a() == nullptr, "");
}

namespace Variadic {
  struct S { int a; bool b; };

  constexpr void variadic_function(int a, ...) {}
  constexpr int f1() {
    variadic_function(1, S{'a', false});
    return 1;
  }
  static_assert(f1() == 1, "");

  constexpr int variadic_function2(...) {
    return 12;
  }
  static_assert(variadic_function2() == 12, "");
  static_assert(variadic_function2(1, 2, 3, 4, 5) == 12, "");
  static_assert(variadic_function2(1, variadic_function2()) == 12, "");

  constexpr int (*VFP)(...) = variadic_function2;
  static_assert(VFP() == 12, "");

  /// Member functions
  struct Foo {
    int a = 0;
    constexpr void bla(...) {}
    constexpr S bla2(...) {
      return S{12, true};
    }
    constexpr Foo(...) : a(1337) {}
    constexpr Foo(void *c, bool b, void*p, ...) : a('a' + b) {}
    constexpr Foo(int a, const S* s, ...) : a(a) {}
  };

  constexpr int foo2() {
    Foo f(1, nullptr);
    auto s = f.bla2(1, 2, S{1, false});
    return s.a + s.b;
  }
  static_assert(foo2() == 13, "");

  constexpr Foo _f = 123;
  static_assert(_f.a == 1337, "");

  constexpr Foo __f(nullptr, false, nullptr, nullptr, 'a', Foo());
  static_assert(__f.a ==  'a', "");


#if __cplusplus >= 202002L
namespace VariadicVirtual {
  class A {
  public:
    constexpr virtual void foo(int &a, ...) {
      a = 1;
    }
  };

  class B : public A {
  public:
    constexpr void foo(int &a, ...) override {
      a = 2;
    }
  };

  constexpr int foo() {
    B b;
    int a;
    b.foo(a, 1,2,nullptr);
    return a;
  }
  static_assert(foo() == 2, "");
} // VariadicVirtual

namespace VariadicQualified {
  class A {
      public:
      constexpr virtual int foo(...) const {
          return 5;
      }
  };
  class B : public A {};
  class C : public B {
      public:
      constexpr int foo(...) const override {
          return B::foo(1,2,3); // B doesn't have a foo(), so this should call A::foo().
      }
      constexpr int foo2() const {
        return this->A::foo(1,2,3,this);
      }
  };
  constexpr C c;
  static_assert(c.foo() == 5);
  static_assert(c.foo2() == 5);
} // VariadicQualified
#endif

}

namespace Packs {
  template<typename...T>
  constexpr int foo() { return sizeof...(T); }
  static_assert(foo<int, char>() == 2, "");
  static_assert(foo<>() == 0, "");
}

namespace AddressOf {
  struct S {} s;
  static_assert(__builtin_addressof(s) == &s, "");

  struct T { constexpr T *operator&() const { return nullptr; } int n; } t;
  constexpr T *pt = __builtin_addressof(t);
  static_assert(&pt->n == &t.n, "");

  struct U { int n : 5; } u;
  int *pbf = __builtin_addressof(u.n); // both-error {{address of bit-field requested}}

  S *ptmp = __builtin_addressof(S{}); // both-error {{taking the address of a temporary}} \
                                      // both-warning {{temporary whose address is used as value of local variable 'ptmp' will be destroyed at the end of the full-expression}}

  constexpr int foo() {return 1;}
  static_assert(__builtin_addressof(foo) == foo, "");

  constexpr _Complex float F = {3, 4};
  static_assert(__builtin_addressof(F) == &F, "");
}

namespace std {
template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };
template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}
}
/// The std::move declaration above gets translated to a builtin function.
namespace Move {
#if __cplusplus >= 202002L
  consteval int f_eval() { // both-note 12{{declared here}}
    return 0;
  }

  /// From test/SemaCXX/cxx2a-consteval.
  struct Copy {
    int(*ptr)();
    constexpr Copy(int(*p)() = nullptr) : ptr(p) {}
    consteval Copy(const Copy&) = default;
  };

  constexpr const Copy &to_lvalue_ref(const Copy &&a) {
    return a;
  }

  void test() {
    constexpr const Copy C;
    // there is no the copy constructor call when its argument is a prvalue because of garanteed copy elision.
    // so we need to test with both prvalue and xvalues.
    { Copy c(C); }
    { Copy c((Copy(&f_eval))); } // both-error {{cannot take address of consteval}}
    { Copy c(std::move(C)); }
    { Copy c(std::move(Copy(&f_eval))); } // both-error {{is not a constant expression}} \
                                          // both-note {{to a consteval}}
    { Copy c(to_lvalue_ref((Copy(&f_eval)))); } // both-error {{is not a constant expression}} \
                                                // both-note {{to a consteval}}
    { Copy c(to_lvalue_ref(std::move(C))); }
    { Copy c(to_lvalue_ref(std::move(Copy(&f_eval)))); } // both-error {{is not a constant expression}} \
                                                         // both-note {{to a consteval}}
    { Copy c = Copy(C); }
    { Copy c = Copy(Copy(&f_eval)); } // both-error {{cannot take address of consteval}}
    { Copy c = Copy(std::move(C)); }
    { Copy c = Copy(std::move(Copy(&f_eval))); } // both-error {{is not a constant expression}} \
                                                 // both-note {{to a consteval}}
    { Copy c = Copy(to_lvalue_ref(Copy(&f_eval))); } // both-error {{is not a constant expression}} \
                                                     // both-note {{to a consteval}}
    { Copy c = Copy(to_lvalue_ref(std::move(C))); }
    { Copy c = Copy(to_lvalue_ref(std::move(Copy(&f_eval)))); } // both-error {{is not a constant expression}} \
                                                                // both-note {{to a consteval}}
    { Copy c; c = Copy(C); }
    { Copy c; c = Copy(Copy(&f_eval)); } // both-error {{cannot take address of consteval}}
    { Copy c; c = Copy(std::move(C)); }
    { Copy c; c = Copy(std::move(Copy(&f_eval))); } // both-error {{is not a constant expression}} \
                                                    // both-note {{to a consteval}}
    { Copy c; c = Copy(to_lvalue_ref(Copy(&f_eval))); } // both-error {{is not a constant expression}} \
                                                        // both-note {{to a consteval}}
    { Copy c; c = Copy(to_lvalue_ref(std::move(C))); }
    { Copy c; c = Copy(to_lvalue_ref(std::move(Copy(&f_eval)))); } // both-error {{is not a constant expression}} \
                                                                   // both-note {{to a consteval}}
  }
#endif
  constexpr int A = std::move(5);
  static_assert(A == 5, "");
}

namespace StaticLocals {
  void test() {
    static int j; // both-note {{declared here}}
    static_assert(&j != nullptr, ""); // both-warning {{always true}}

    static_assert(j == 0, ""); // both-error {{not an integral constant expression}} \
                               // both-note {{read of non-const variable 'j'}}

    static int k = 0; // both-note {{declared here}}
    static_assert(k == 0, ""); // both-error {{not an integral constant expression}} \
                               // both-note {{read of non-const variable 'k'}}

    static const int l = 12;
    static_assert(l == 12, "");

    static const int m; // both-error {{default initialization}}
    static_assert(m == 0, "");
  }
}

namespace Local {
  /// We used to run into infinite recursin here because we were
  /// trying to evaluate t's initializer while evaluating t's initializer.
  int a() {
    const int t=t;
    return t;
  }
}

namespace VariadicOperator {
  struct Callable {
    float& operator()(...);
  };

  void test_callable(Callable c) {
    float &fr = c(10);
  }
}
