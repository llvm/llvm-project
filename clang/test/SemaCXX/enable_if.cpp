// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s
typedef int (*fp)(int);
int surrogate(int);
struct Incomplete;  // expected-note{{forward declaration of 'Incomplete'}} \
                    // expected-note {{forward declaration of 'Incomplete'}}

struct X { // expected-note{{defined here}}
  X() = default;  // expected-note{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  X(const X&) = default;  // expected-note{{candidate constructor not viable: no known conversion from 'bool' to 'const X' for 1st argument}}
  X(bool b) __attribute__((enable_if(b, "chosen when 'b' is true")));  // expected-note{{candidate disabled: chosen when 'b' is true}}

  void f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));
  void f(int n) __attribute__((enable_if(n == 1, "chosen when 'n' is one")));  // expected-note{{member declaration nearly matches}} expected-note 2{{candidate disabled: chosen when 'n' is one}}

  void g(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));  // expected-note{{candidate disabled: chosen when 'n' is zero}}

  void h(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}

  static void s(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));  // expected-note2{{candidate disabled: chosen when 'n' is zero}}

  void conflict(int n) __attribute__((enable_if(n+n == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}
  void conflict(int n) __attribute__((enable_if(n*2 == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}

  void hidden_by_argument_conversion(Incomplete n, int m = 0) __attribute__((enable_if(m == 10, "chosen when 'm' is ten")));
  Incomplete hidden_by_incomplete_return_value(int n = 0) __attribute__((enable_if(n == 10, "chosen when 'n' is ten"))); // expected-note{{'hidden_by_incomplete_return_value' declared here}}

  operator long() __attribute__((enable_if(true, "chosen on your platform")));
  operator int() __attribute__((enable_if(false, "chosen on other platform")));

  operator fp() __attribute__((enable_if(false, "never enabled"))) { return surrogate; }  // expected-note{{conversion candidate of type 'int (*)(int)'}}  // FIXME: the message is not displayed
};

void X::f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")))  // expected-note{{member declaration nearly matches}} expected-note 2{{candidate disabled: chosen when 'n' is zero}}
{
}

void X::f(int n) __attribute__((enable_if(n == 2, "chosen when 'n' is two")))  // expected-error{{out-of-line definition of 'f' does not match any declaration in 'X'}}
{
}

X x1(true);
X x2(false); // expected-error{{no matching constructor for initialization of 'X'}}

__attribute__((deprecated)) constexpr int old() { return 0; }  // expected-note2{{'old' has been explicitly marked deprecated here}}
void deprec1(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}
void deprec2(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}

void overloaded(int);
void overloaded(long);

struct Int {
  constexpr Int(int i) : i(i) { }
  constexpr operator int() const { return i; }
  int i;
};

void default_argument(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}
void default_argument_promotion(int n, int m = Int(0)) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}

struct Nothing { };
template<typename T> void typedep(T t) __attribute__((enable_if(t, "")));  // expected-note{{candidate disabled:}}  expected-error{{value of type 'Nothing' is not contextually convertible to 'bool'}}
template<int N> void valuedep() __attribute__((enable_if(N == 1, "")));

// FIXME: we skip potential constant expression evaluation on value dependent
// enable-if expressions
int not_constexpr();
template<int N> void valuedep() __attribute__((enable_if(N == not_constexpr(), "")));

template <typename T> void instantiationdep() __attribute__((enable_if(sizeof(sizeof(T)) != 0, "")));

void test() {
  X x;
  x.f(0);
  x.f(1);
  x.f(2);  // expected-error{{no matching member function for call to 'f'}}
  x.f(3);  // expected-error{{no matching member function for call to 'f'}}

  x.g(0);
  x.g(1);  // expected-error{{no matching member function for call to 'g'}}

  x.h(0);
  x.h(1, 2);  // expected-error{{no matching member function for call to 'h'}}

  x.s(0);
  x.s(1);  // expected-error{{no matching member function for call to 's'}}

  X::s(0);
  X::s(1);  // expected-error{{no matching member function for call to 's'}}

  x.conflict(5);  // expected-error{{call to member function 'conflict' is ambiguous}}

  x.hidden_by_argument_conversion(10);  // expected-error{{argument type 'Incomplete' is incomplete}}
  x.hidden_by_incomplete_return_value(10);  // expected-error{{calling 'hidden_by_incomplete_return_value' with incomplete return type 'Incomplete'}}

  deprec2(0);

  overloaded(x);

  default_argument(0);
  default_argument(1, 2);  // expected-error{{no matching function for call to 'default_argument'}}

  default_argument_promotion(0);
  default_argument_promotion(1, 2);  // expected-error{{no matching function for call to 'default_argument_promotion'}}

  int i = x(1);  // expected-error{{no matching function for call to object of type 'X'}}

  Nothing n;
  typedep(0);  // expected-error{{no matching function for call to 'typedep'}}
  typedep(1);
  typedep(n);  // expected-note{{in instantiation of function template specialization 'typedep<Nothing>' requested here}}
}

template <typename T> class C {
  void f() __attribute__((enable_if(T::expr == 0, ""))) {}
  void g() { f(); }
};

int fn3(bool b) __attribute__((enable_if(b, ""))); // FIXME: This test should net 0 error messages.
template <class T> void test3() {
  fn3(sizeof(T) == 1); // expected-error{{no matching function for call to 'fn3'}} expected-note@-2{{candidate disabled}}
}

template <typename T>
struct Y {
  T h(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}
};

void test4() {
  Y<int> y;

  int t0 = y.h(0);
  int t1 = y.h(1, 2);  // expected-error{{no matching member function for call to 'h'}}
}

// FIXME: issue an error (without instantiation) because ::h(T()) is not
// convertible to bool, because return types aren't overloadable.
void h(int);
template <typename T> void outer() {
  void local_function() __attribute__((enable_if(::h(T()), "")));
  local_function(); // expected-error{{no matching function for call to 'local_function'}} expected-note@-1{{candidate disabled}}
};

namespace PR20988 {
  struct Integer {
    Integer(int);
  };

  int fn1(const Integer &) __attribute__((enable_if(true, "")));
  template <class T> void test1() {
    int &expr = T::expr();
    fn1(expr);
  }

  int fn2(const Integer &) __attribute__((enable_if(false, "")));  // expected-note{{candidate disabled}}
  template <class T> void test2() {
    int &expr = T::expr();
    fn2(expr);  // expected-error{{no matching function for call to 'fn2'}}
  }

  int fn3(bool b) __attribute__((enable_if(b, ""))); // FIXME: This test should net 0 error messages.
  template <class T> void test3() {
    fn3(sizeof(T) == 1); // expected-error{{no matching function for call to 'fn3'}} expected-note@-2{{candidate disabled}}
  }
}

namespace FnPtrs {
  int ovlFoo(int m) __attribute__((enable_if(m > 0, "")));
  int ovlFoo(int m);

  void test() {
    // Assignment gives us a different code path than declarations, and `&foo`
    // gives us a different code path than `foo`
    int (*p)(int) = ovlFoo;
    int (*p2)(int) = &ovlFoo;
    int (*a)(int);
    a = ovlFoo;
    a = &ovlFoo;
  }

  int ovlBar(int) __attribute__((enable_if(true, "")));
  int ovlBar(int m) __attribute__((enable_if(false, "")));
  void test2() {
    int (*p)(int) = ovlBar;
    int (*p2)(int) = &ovlBar;
    int (*a)(int);
    a = ovlBar;
    a = &ovlBar;
  }

  int ovlConflict(int m) __attribute__((enable_if(true, "")));
  int ovlConflict(int m) __attribute__((enable_if(1, "")));
  void test3() {
    int (*p)(int) = ovlConflict; // expected-error{{address of overloaded function 'ovlConflict' is ambiguous}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    int (*p2)(int) = &ovlConflict; // expected-error{{address of overloaded function 'ovlConflict' is ambiguous}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    int (*a)(int);
    a = ovlConflict; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    a = &ovlConflict; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
  }

  template <typename T>
  T templated(T m) __attribute__((enable_if(true, ""))) { return T(); }
  template <typename T>
  T templated(T m) __attribute__((enable_if(false, ""))) { return T(); }
  void test4() {
    int (*p)(int) = templated<int>;
    int (*p2)(int) = &templated<int>;
    int (*a)(int);
    a = templated<int>;
    a = &templated<int>;
  }

  template <typename T>
  T templatedBar(T m) __attribute__((enable_if(m > 0, ""))) { return T(); }
  void test5() {
    int (*p)(int) = templatedBar<int>; // expected-error{{address of overloaded function 'templatedBar' does not match required type 'int (int)'}} expected-note@214{{candidate function made ineligible by enable_if}}
    int (*p2)(int) = &templatedBar<int>; // expected-error{{address of overloaded function 'templatedBar' does not match required type 'int (int)'}} expected-note@214{{candidate function made ineligible by enable_if}}
    int (*a)(int);
    a = templatedBar<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@214{{candidate function made ineligible by enable_if}}
    a = &templatedBar<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@214{{candidate function made ineligible by enable_if}}
  }

  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(false, ""))) { return T(); }
  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(true, ""))) { return T(); }
  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(1, ""))) { return T(); }
  void test6() {
    int (*p)(int) = templatedConflict<int>; // expected-error{{address of overloaded function 'templatedConflict' is ambiguous}} expected-note@224{{candidate function made ineligible by enable_if}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    int (*p0)(int) = &templatedConflict<int>; // expected-error{{address of overloaded function 'templatedConflict' is ambiguous}} expected-note@224{{candidate function made ineligible by enable_if}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    int (*a)(int);
    a = templatedConflict<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    a = &templatedConflict<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
  }

  int ovlNoCandidate(int m) __attribute__((enable_if(false, "")));
  int ovlNoCandidate(int m) __attribute__((enable_if(0, "")));
  void test7() {
    int (*p)(int) = ovlNoCandidate; // expected-error{{address of overloaded function 'ovlNoCandidate' does not match required type}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    int (*p2)(int) = &ovlNoCandidate; // expected-error{{address of overloaded function 'ovlNoCandidate' does not match required type}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    int (*a)(int);
    a = ovlNoCandidate; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    a = &ovlNoCandidate; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
  }

  int noOvlNoCandidate(int m) __attribute__((enable_if(false, "")));
  void test8() {
    int (*p)(int) = noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' because it has one or more non-tautological enable_if conditions}}
    int (*p2)(int) = &noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' because it has one or more non-tautological enable_if conditions}}
    int (*a)(int);
    a = noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' because it has one or more non-tautological enable_if conditions}}
    a = &noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' because it has one or more non-tautological enable_if conditions}}
  }
}

namespace casting {
using VoidFnTy = void (*)();

void foo(void *c) __attribute__((enable_if(0, "")));
void foo(int *c) __attribute__((enable_if(c, "")));
void foo(char *c) __attribute__((enable_if(1, "")));

void testIt() {
  auto A = reinterpret_cast<VoidFnTy>(foo);
  auto AAmp = reinterpret_cast<VoidFnTy>(&foo);

  using VoidFooTy = void (*)(void *);
  auto B = reinterpret_cast<VoidFooTy>(foo);
  auto BAmp = reinterpret_cast<VoidFooTy>(&foo);

  using IntFooTy = void (*)(int *);
  auto C = reinterpret_cast<IntFooTy>(foo);
  auto CAmp = reinterpret_cast<IntFooTy>(&foo);

  using CharFooTy = void (*)(void *);
  auto D = reinterpret_cast<CharFooTy>(foo);
  auto DAmp = reinterpret_cast<CharFooTy>(&foo);
}

void testItCStyle() {
  auto A = (VoidFnTy)foo;
  auto AAmp = (VoidFnTy)&foo;

  using VoidFooTy = void (*)(void *);
  auto B = (VoidFooTy)foo;
  auto BAmp = (VoidFooTy)&foo;

  using IntFooTy = void (*)(int *);
  auto C = (IntFooTy)foo;
  auto CAmp = (IntFooTy)&foo;

  using CharFooTy = void (*)(void *);
  auto D = (CharFooTy)foo;
  auto DAmp = (CharFooTy)&foo;
}
}

namespace casting_templates {
template <typename T> void foo(T) {} // expected-note 4 {{candidate function}}

void foo(int *c) __attribute__((enable_if(c, ""))); //expected-note 4 {{candidate function}}
void foo(char *c) __attribute__((enable_if(c, ""))); //expected-note 4 {{candidate function}}

void testIt() {
  using IntFooTy = void (*)(int *);
  auto A = reinterpret_cast<IntFooTy>(foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}
  auto ARef = reinterpret_cast<IntFooTy>(&foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}
  auto AExplicit = reinterpret_cast<IntFooTy>(foo<int*>);

  using CharFooTy = void (*)(char *);
  auto B = reinterpret_cast<CharFooTy>(foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}
  auto BRef = reinterpret_cast<CharFooTy>(&foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}
  auto BExplicit = reinterpret_cast<CharFooTy>(foo<char*>);
}

void testItCStyle() {
  // constexpr is usable here because all of these should become static_casts.
  using IntFooTy = void (*)(int *);
  constexpr auto A = (IntFooTy)foo;
  constexpr auto ARef = (IntFooTy)&foo;
  constexpr auto AExplicit = (IntFooTy)foo<int*>;

  using CharFooTy = void (*)(char *);
  constexpr auto B = (CharFooTy)foo;
  constexpr auto BRef = (CharFooTy)&foo;
  constexpr auto BExplicit = (CharFooTy)foo<char*>;

  static_assert(A == ARef && ARef == AExplicit, "");
  static_assert(B == BRef && BRef == BExplicit, "");
}
}

namespace multiple_matches {
using NoMatchTy = void (*)();

void foo(float *c); //expected-note 4 {{candidate function}}
void foo(int *c) __attribute__((enable_if(1, ""))); //expected-note 4 {{candidate function}}
void foo(char *c) __attribute__((enable_if(1, ""))); //expected-note 4 {{candidate function}}

void testIt() {
  auto A = reinterpret_cast<NoMatchTy>(foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}
  auto ARef = reinterpret_cast<NoMatchTy>(&foo); // expected-error{{reinterpret_cast cannot resolve overloaded function 'foo' to type}}

  auto C = (NoMatchTy)foo; // expected-error{{address of overloaded function 'foo' does not match required type 'void ()'}}
  auto CRef = (NoMatchTy)&foo; // expected-error{{address of overloaded function 'foo' does not match required type 'void ()'}}
}
}

namespace PR27122 {
// (slightly reduced) code that motivated the bug...
namespace ns {
void Function(int num)
  __attribute__((enable_if(num != 0, "")));
void Function(int num, int a0)
  __attribute__((enable_if(num != 1, "")));
}  // namespace ns

using ns::Function; // expected-note 3{{declared here}}
void Run() {
  Functioon(0); // expected-error{{use of undeclared identifier}} expected-error{{too few arguments}}
  Functioon(0, 1); // expected-error{{use of undeclared identifier}}
  Functioon(0, 1, 2); // expected-error{{use of undeclared identifier}}
}

// Extra tests
void regularEnableIf(int a) __attribute__((enable_if(a, ""))); // expected-note 3{{declared here}} expected-note 3{{candidate function not viable}}
void runRegularEnableIf() {
  regularEnableIf(0, 2); // expected-error{{no matching function}}
  regularEnableIf(1, 2); // expected-error{{no matching function}}
  regularEnableIf(); // expected-error{{no matching function}}

  // Test without getting overload resolution involved
  ::PR27122::regularEnableIf(0, 2); // expected-error{{too many arguments}}
  ::PR27122::regularEnableIf(1, 2); // expected-error{{too many arguments}}
  ::PR27122::regularEnableIf(); // expected-error{{too few arguments}}
}

struct Foo {
  void bar(int i) __attribute__((enable_if(i, ""))); // expected-note 2{{declared here}}
};

void runFoo() {
  Foo f;
  f.bar(); // expected-error{{too few arguments}}
  f.bar(1, 2); // expected-error{{too many arguments}}
}
}

// Ideally, we should be able to handle value-dependent expressions sanely.
// Sadly, that isn't the case at the moment.
namespace dependent {
int error(int N) __attribute__((enable_if(N, ""))); // expected-note{{candidate disabled}}
int error(int N) __attribute__((enable_if(!N, ""))); // expected-note{{candidate disabled}}
template <int N> int callUnavailable() {
  return error(N); // expected-error{{no matching function for call to 'error'}}
}

constexpr int noError(int N) __attribute__((enable_if(N, ""))) { return -1; }
constexpr int noError(int N) __attribute__((enable_if(!N, ""))) { return -1; }
constexpr int noError(int N) { return 0; }

template <int N>
constexpr int callNoError() { return noError(N); }
static_assert(callNoError<0>() == 0, "");
static_assert(callNoError<1>() == 0, "");

template <int N> constexpr int templated() __attribute__((enable_if(N, ""))) {
  return 1;
}

constexpr int A = templated<0>(); // expected-error{{no matching function for call to 'templated'}} expected-note@-4{{candidate disabled}}
static_assert(templated<1>() == 1, "");

template <int N> constexpr int callTemplated() { return templated<N>(); }

constexpr int B = 10 +                // expected-error {{initialized by a constant expression}}
                  callTemplated<0>(); // expected-error@-3{{no matching function for call to 'templated'}} expected-note{{in instantiation of function template}} expected-note@-10{{candidate disabled}}
static_assert(callTemplated<1>() == 1, "");
}

namespace variadic {
void foo(int a, int b = 0, ...) __attribute__((enable_if(a && b, ""))); // expected-note 6{{disabled}}

void testFoo() {
  foo(1, 1);
  foo(1, 1, 2);
  foo(1, 1, 2, 3);

  foo(1, 0); // expected-error{{no matching}}
  foo(1, 0, 2); // expected-error{{no matching}}
  foo(1, 0, 2, 3); // expected-error{{no matching}}

  int m;
  foo(1, 1);
  foo(1, 1, m);
  foo(1, 1, m, 3);

  foo(1, 0); // expected-error{{no matching}}
  foo(1, 0, m); // expected-error{{no matching}}
  foo(1, 0, m, 3); // expected-error{{no matching}}
}
}

// Tests that we emit errors at the point of the method call, rather than the
// beginning of the expression that happens to be a member call.
namespace member_loc {
  struct Foo { void bar() __attribute__((enable_if(0, ""))); }; // expected-note{{disabled}}
  void testFoo() {
    Foo()
      .bar(); // expected-error{{no matching member function}}
  }
}

// Prior bug: we wouldn't properly convert conditions to bools when
// instantiating templates in some cases.
namespace template_instantiation {
template <typename T>
struct Foo {
  void bar(int a) __attribute__((enable_if(a, ""))); // expected-note{{disabled}}
};

void runFoo() {
  Foo<double>().bar(0); // expected-error{{no matching}}
  Foo<double>().bar(1);
}
}

namespace instantiate_constexpr_in_enable_if {
  template<typename T> struct X {
    static constexpr bool ok() { return true; }
    void f() __attribute__((enable_if(ok(), "")));
  };
  void g() { X<int>().f(); }
}

namespace PR31934 {
int foo(int a) __attribute__((enable_if(a, "")));
int runFn(int (&)(int));

void run() {
  {
    int (&bar)(int) = foo; // expected-error{{cannot take address of function 'foo'}}
    int baz = runFn(foo); // expected-error{{cannot take address of function 'foo'}}
  }

  {
    int (&bar)(int) = (foo); // expected-error{{cannot take address of function 'foo'}}
    int baz = runFn((foo)); // expected-error{{cannot take address of function 'foo'}}
  }

  {
    int (&bar)(int) = static_cast<int (&)(int)>(foo); // expected-error{{cannot take address of function 'foo'}}
    int baz = runFn(static_cast<int (&)(int)>(foo)); // expected-error{{cannot take address of function 'foo'}}
  }

  {
    int (&bar)(int) = static_cast<int (&)(int)>((foo)); // expected-error{{cannot take address of function 'foo'}}
    int baz = runFn(static_cast<int (&)(int)>((foo))); // expected-error{{cannot take address of function 'foo'}}
  }
}
}

namespace TypeOfFn {
  template <typename T, typename U>
  struct is_same;

  template <typename T> struct is_same<T, T> {
    enum { value = 1 };
  };

  void foo(int a) __attribute__((enable_if(a, "")));
  void foo(float a) __attribute__((enable_if(1, "")));

  static_assert(is_same<__typeof__(foo)*, decltype(&foo)>::value, "");
}

namespace InConstantContext {
void foo(const char *s) __attribute__((enable_if(((void)__builtin_constant_p(*s), true), "trap"))) {}

void test() {
  InConstantContext::foo("abc");
}
} // namespace InConstantContext

namespace StringLiteralDetector {
  void need_string_literal(const char *p) __attribute__((enable_if(__builtin_constant_p(p), "argument is not a string literal"))); // expected-note 2{{not a string literal}}
  void test(const char *unknown) {
    need_string_literal("foo");
    need_string_literal(unknown); // expected-error {{no matching function}}
    constexpr char str[] = "bar";
    need_string_literal(str); // expected-error {{no matching function}}
  }
}

namespace IgnoreUnusedArgSideEffects {
  struct A { ~A(); };
  void f(A a, bool b) __attribute__((enable_if(b, ""))); // expected-note 2-3{{disabled}}
  void test() {
    f(A(), true);
    f(A(), false); // expected-error {{no matching function}}
    int n;
    f((n = 1, A()), true);
    f(A(), (n = 1, true)); // expected-error {{no matching function}}
    f(A(), (A(), true));
  }

#if __cplusplus > 201702L
  struct B { constexpr ~B() {} bool b; };
  void g(B b) __attribute__((enable_if(b.b, ""))); // expected-note {{disabled}}
  void test2() {
    g(B{true});
    g(B{false}); // expected-error {{no matching function}}
    f(A(), B{true}.b);
    f(A(), B{false}.b); // expected-error {{no matching function}}
  }

  // First condition is non-constant due to non-constexpr destructor of A.
  int &h() __attribute__((enable_if((A(), true), "")));
  float &h() __attribute__((enable_if((B(), true), "")));
  float &x = h();
#endif
}

namespace DefaultArgs {
  void f(int n = __builtin_LINE()) __attribute__((enable_if(n == 12345, "only callable on line 12345"))); // expected-note {{only callable on line 12345}}
  void g() { f(); } // expected-error {{no matching function}}
#line 12345
  void h() { f(); }

  template<typename T> void x(int n = T()) __attribute__((enable_if(n == 0, ""))) {} // expected-note {{candidate}}
  void y() { x<int>(); }
  struct Z { constexpr operator int() const { return 1; } };
  void z() { x<Z>(); } // expected-error {{no matching function}}
}
