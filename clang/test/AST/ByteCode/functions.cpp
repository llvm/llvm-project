// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -pedantic -verify=expected,both %s
// RUN: %clang_cc1 -std=c++14 -fexperimental-new-constant-interpreter -pedantic -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -pedantic -verify=expected,both %s
// RUN: %clang_cc1 -pedantic -verify=ref,both %s
// RUN: %clang_cc1 -pedantic -std=c++14 -verify=ref,both %s
// RUN: %clang_cc1 -pedantic -std=c++20 -verify=ref,both %s

#define fold(x) (__builtin_constant_p(0) ? (x) : (x))

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


namespace ToBool {
  void mismatched(int x) {}
  typedef void (*callback_t)(int);
  void foo() {
    callback_t callback = (callback_t)mismatched; // warns
    /// Casts a function pointer to a boolean and then back to a function pointer.
    /// This is extracted from test/Sema/callingconv-cast.c
    callback = (callback_t)!mismatched; // both-warning {{address of function 'mismatched' will always evaluate to 'true'}} \
                                        // both-note {{prefix with the address-of operator to silence this warning}}
  }
}


}

namespace Comparison {
  void f(), g();
  constexpr void (*pf)() = &f, (*pg)() = &g;

  constexpr bool u13 = pf < pg; // both-warning {{ordered comparison of function pointers}} \
                                // both-error {{must be initialized by a constant expression}} \
                                // both-note {{comparison between pointers to unrelated objects '&f' and '&g' has unspecified value}}

  constexpr bool u14 = pf < (void(*)())nullptr; // both-warning {{ordered comparison of function pointers}} \
                                                // both-error {{must be initialized by a constant expression}} \
                                                // both-note {{comparison between pointers to unrelated objects '&f' and 'nullptr' has unspecified value}}



  static_assert(pf != pg, "");
  static_assert(pf == &f, "");
  static_assert(pg == &g, "");
}

  constexpr int Double(int n) { return 2 * n; }
  constexpr int Triple(int n) { return 3 * n; }
  constexpr int Twice(int (*F)(int), int n) { return F(F(n)); }
  constexpr int Quadruple(int n) { return Twice(Double, n); }
  constexpr auto Select(int n) -> int (*)(int) {
    return n == 2 ? &Double : n == 3 ? &Triple : n == 4 ? &Quadruple : 0;
  }
  constexpr int Apply(int (*F)(int), int n) { return F(n); } // both-note {{'F' evaluates to a null function pointer}}

  constexpr int Invalid = Apply(Select(0), 0); // both-error {{must be initialized by a constant expression}} \
                                               // both-note {{in call to 'Apply(nullptr, 0)'}}
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

  /// FIXME: Both interpreters should diagnose this. We're returning a pointer to a local
  /// variable.
  static_assert(p() == nullptr, ""); // both-error {{static assertion failed}}

  constexpr const int &p2() {
    int a = 12; // both-note {{declared here}}
    return a; // both-warning {{reference to stack memory associated with local variable}}
  }

  static_assert(p2() == 12, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{read of variable whose lifetime has ended}}
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

  constexpr _Complex float F = {3, 4}; // both-warning {{'_Complex' is a C99 extension}}
  static_assert(__builtin_addressof(F) == &F, "");

  void testAddressof(int x) {
    static_assert(&x == __builtin_addressof(x), "");
  }

  struct TS {
    constexpr bool f(TS s) const {
      /// The addressof call has a CXXConstructExpr as a parameter.
      return this != __builtin_addressof(s);
    }
  };
  constexpr bool exprAddressOf() {
    TS s;
    return s.f(s);
  }
  static_assert(exprAddressOf(), "");
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

namespace WeakCompare {
  [[gnu::weak]]void weak_method();
  static_assert(weak_method != nullptr, ""); // both-error {{not an integral constant expression}} \
                                             // both-note {{comparison against address of weak declaration '&weak_method' can only be performed at runtim}}

  constexpr auto A = &weak_method;
  static_assert(A != nullptr, ""); // both-error {{not an integral constant expression}} \
                                   // both-note {{comparison against address of weak declaration '&weak_method' can only be performed at runtim}}
}

namespace FromIntegral {
#if __cplusplus >= 202002L
  typedef double (*DoubleFn)();
  int a[(int)DoubleFn((void*)-1)()]; // both-error {{not allowed at file scope}} \
                                    // both-warning {{variable length arrays}}
  int b[(int)DoubleFn((void*)(-1 + 1))()]; // both-error {{not allowed at file scope}} \
                                           // both-note {{evaluates to a null function pointer}} \
                                           // both-warning {{variable length arrays}}
#endif
}

namespace {
  template <typename T> using id = T;
  template <typename T>
  constexpr void g() {
    constexpr id<void (T)> f;
  }

  static_assert((g<int>(), true), "");
}

namespace {
  /// The InitListExpr here is of void type.
  void bir [[clang::annotate("B", {1, 2, 3, 4})]] (); // both-error {{'clang::annotate' attribute requires parameter 1 to be a constant expression}} \
                                                      // both-note {{subexpression not valid in a constant expression}}
}

namespace FuncPtrParam {
  void foo(int(&a)()) {
    *a; // both-warning {{expression result unused}}
  }
}

namespace {
  void f() noexcept;
  void (&r)() = f;
  void (&cond3)() = r;
}

namespace FunctionCast {
  // When folding, we allow functions to be cast to different types. We only
  // allow calls if the dynamic type of the pointer matches the type of the
  // call.
  constexpr int f() { return 1; }
  constexpr void* f2() { return nullptr; }
  constexpr int f3(int a) { return a; }
  typedef double (*DoubleFn)();
  typedef int (*IntFn)();
  typedef int* (*IntPtrFn)();
  constexpr int test1 = (int)DoubleFn(f)(); // both-error {{constant expression}} both-note {{reinterpret_cast}}
  // FIXME: We should print a note explaining the error.
  constexpr int test2 = (int)fold(DoubleFn(f))(); // both-error {{constant expression}}
  constexpr int test3 = (int)IntFn(f)();    // no-op cast
  constexpr int test4 = fold(IntFn(DoubleFn(f)))();
  constexpr int test5 = IntFn(fold(DoubleFn(f)))(); // both-error {{constant expression}} \
                                                    // both-note {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}
  // FIXME: Interpreter is less strict here.
  constexpr int test6 = fold(IntPtrFn(f2))() == nullptr; // ref-error {{constant expression}}
  // FIXME: The following crashes interpreter
  // constexpr int test6 = fold(IntFn(f3)());
}

#if __cplusplus >= 202002L
namespace StableAddress {
  template<unsigned N> struct str {
    char arr[N];
  };
  // FIXME: Deduction guide not needed with P1816R0.
  template<unsigned N> str(const char (&)[N]) -> str<N>;

  template<str s> constexpr int sum() {
    int n = 0;
    for (char c : s.arr)
      n += c;
    return n;
  }
  static_assert(sum<str{"$hello $world."}>() == 1234, "");
}
#endif

namespace NoDiags {
  void huh();
  template <unsigned>
  constexpr void hd_fun() {
    huh();
  }

  constexpr bool foo() {
    hd_fun<1>();
    return true;
  }
}

namespace EnableIfWithTemporary {
  struct A { ~A(); };
  int &h() __attribute__((enable_if((A(), true), ""))); // both-warning {{clang extension}}
}

namespace LocalVarForParmVarDecl {
  struct Iter {
    void *p;
  };
  constexpr bool bar2(Iter A) {
    return true;
  }
  constexpr bool bar(Iter A, bool b) {
    if (b)
      return true;

    return bar(A, true);
  }
  constexpr int foo() {
    return bar(Iter(), false);
  }
  static_assert(foo(), "");
}

namespace PtrPtrCast {
  void foo() { ; }
  void bar(int *a) { a = (int *)(void *)(foo); }
}
