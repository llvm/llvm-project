// RUN: %clang_cc1 -std=c++23  -x c++ %s -verify
// RUN: %clang_cc1 -std=c++20 -pedantic -x c++ %s -verify=ext,expected
// RUN: %clang_cc1 -std=c++23  -x c++ %s -verify -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++20 -pedantic -x c++ %s -verify=ext,expected -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++26  -x c++ %s -verify
// RUN: %clang_cc1 -std=c++26  -x c++ %s -verify -fexperimental-new-constant-interpreter

struct A{};
struct B{ explicit operator bool() { return true; } };

// This should be the first test case of this file.
void IsActOnFinishFullExprCalled() {
  // Do not add other test cases to this function.
  // Make sure `ActOnFinishFullExpr` is called and creates `ExprWithCleanups`
  // to avoid assertion failure.
  [[assume(B{})]]; // ext-warning {{C++23 extension}}
}

template <bool cond>
void f() {
  [[assume(cond)]]; // ext-warning {{C++23 extension}}
}

template <bool cond>
struct S {
  void f() {
    [[assume(cond)]]; // ext-warning {{C++23 extension}}
  }

  template <typename T>
  constexpr bool g() {
    [[assume(cond == sizeof(T))]]; // expected-note {{assumption evaluated to false}} ext-warning {{C++23 extension}}
    return true;
  }
};

bool f2();

template <typename T>
constexpr void f3() {
  [[assume(T{})]]; // expected-error {{not contextually convertible to 'bool'}} ext-warning {{C++23 extension}}
}

void g(int x) {
  f<true>();
  f<false>();
  S<true>{}.f();
  S<false>{}.f();
  S<true>{}.g<char>();
  S<true>{}.g<int>();
  [[assume(f2())]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}

  [[assume((x = 3))]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} // ext-warning {{C++23 extension}}
  [[assume(x++)]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} // ext-warning {{C++23 extension}}
  [[assume(++x)]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} // ext-warning {{C++23 extension}}
  [[assume([]{ return true; }())]]; // ext-warning {{C++23 extension}}
  [[assume(B{})]]; // ext-warning {{C++23 extension}}
  [[assume((1, 2))]]; // expected-warning {{has no effect}} // ext-warning {{C++23 extension}}

  f3<A>(); // expected-note {{in instantiation of}}
  f3<B>();
  [[assume]]; // expected-error {{takes one argument}}
  [[assume(z)]]; // expected-error {{undeclared identifier}}
  [[assume(A{})]]; // expected-error {{not contextually convertible to 'bool'}}
  [[assume(true)]] if (true) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] for (;false;) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] while (false) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] label:; // expected-error {{cannot be applied to a declaration}}
  [[assume(true)]] goto label; // expected-error {{only applies to empty statements}}

  // Also check variant spellings.
  __attribute__((__assume__(true))); // Should not issue a warning because it doesn't use the [[]] spelling.
  __attribute__((assume(true))) {}; // expected-error {{only applies to empty statements}}
  [[clang::assume(true)]] {}; // expected-error {{only applies to empty statements}}
}

// Check that 'x' is ODR-used here.
constexpr int h(int x) { return sizeof([=] { [[assume(x)]]; }); } // ext-warning {{C++23 extension}}
static_assert(h(4) == sizeof(int));

static_assert(__has_cpp_attribute(assume) == 202207L);
static_assert(__has_attribute(assume));

constexpr bool i() { // ext-error {{never produces a constant expression}}
  [[assume(false)]]; // ext-note {{assumption evaluated to false}} expected-note {{assumption evaluated to false}} ext-warning {{C++23 extension}}
  return true;
}

constexpr bool j(bool b) {
  [[assume(B{})]]; // ext-warning {{C++23 extension}}
  return true;
}

static_assert(i()); // expected-error {{not an integral constant expression}} expected-note {{in call to}}
static_assert(j(true));
static_assert(j(false));
static_assert(S<true>{}.g<char>());
static_assert(S<false>{}.g<A>()); // expected-error {{not an integral constant expression}} expected-note {{in call to}}


template <typename T>
constexpr bool f4() {
  [[assume(!T{})]]; // expected-error {{invalid argument type 'D'}} ext-warning {{C++23 extension}}
  return sizeof(T) == sizeof(int);
}

template <typename T>
concept C = f4<T>(); // expected-note {{in instantiation of}}
                     // expected-note@-1 {{while substituting}}
                     // expected-error@-2 {{resulted in a non-constant expression}}
                     // expected-note@-3 {{because substituted constraint expression is ill-formed: substitution into constraint expression resulted in a non-constant expression}}

struct D {
  int x;
};

struct E {
  int x;
  constexpr explicit operator bool() { return false; }
};

struct F {
  int x;
  int y;
  constexpr explicit operator bool() { return false; }
};

template <typename T>
constexpr int f5() requires C<T> { return 1; } // expected-note {{while checking the satisfaction}}
                                               // expected-note@-1 {{candidate template ignored}}

template <typename T>
constexpr int f5() requires (!C<T>) { return 2; } // expected-note {{while checking the satisfaction}} \
                                                  // expected-note {{while substituting template arguments}} \
                                                  // expected-note {{candidate template ignored}}

static_assert(f5<int>() == 1);
static_assert(f5<D>() == 1); // expected-note 2 {{while checking constraint satisfaction}}
                             // expected-note@-1 2 {{while substituting deduced template arguments}}
                             // expected-error@-2 {{no matching function for call}}

static_assert(f5<double>() == 2);
static_assert(f5<E>() == 1);
static_assert(f5<F>() == 2);

// Do not validate assumptions whose evaluation would have side-effects.
constexpr int foo() {
  int a = 0;
  [[assume(a++)]] [[assume(++a)]]; // expected-warning 2 {{assumption is ignored because it contains (potential) side-effects}} ext-warning 2 {{C++23 extension}}
  [[assume((a+=1))]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}
  return a;
}

static_assert(foo() == 0);

template <bool ...val>
void f() {
    [[assume(val)]]; // expected-error {{expression contains unexpanded parameter pack}}
}

namespace gh71858 {
int
foo (int x, int y)
{
  __attribute__((assume(x == 42)));
  __attribute__((assume(++y == 43))); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
  return x + y;
}
}

// Do not crash when assumptions are unreachable.
namespace gh106898 {
int foo () {
    while(1);
    int a = 0, b = 1;
    __attribute__((assume (a < b)));
}
}

namespace assume_side_effect_analysis {
bool no_side_effects() { return true; }

bool nested_no_side_effects() { return no_side_effects(); }

bool side_effects(int &x) { return ++x; }

bool nested_side_effects(int &x) { return side_effects(x); }

bool declared_pure() __attribute__((pure));
bool declared_const() __attribute__((const));

bool recursive_b();
bool recursive_a() { return recursive_b(); }
bool recursive_b() { return recursive_a(); }

struct V {
  virtual bool f() { return true; }
};

void test() {
  int x = 0;
  V v;
  V &vr = v;
  bool (*fp)() = no_side_effects;

  [[assume(no_side_effects())]]; // ext-warning {{C++23 extension}}
  [[assume(nested_no_side_effects())]]; // ext-warning {{C++23 extension}}
  [[assume(nested_side_effects(x))]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}
  [[assume(declared_pure())]]; // ext-warning {{C++23 extension}}
  [[assume(declared_const())]]; // ext-warning {{C++23 extension}}
  [[assume(fp())]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}
  [[assume(recursive_a())]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}
  [[assume(vr.f())]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}} ext-warning {{C++23 extension}}
}
} // namespace assume_side_effect_analysis

namespace GH114787 {

// FIXME: Correct the C++26 value
#if __cplusplus >= 202400L

constexpr int test(auto... xs) {
  // FIXME: Investigate why addresses of PackIndexingExprs are printed for the next
  // 'in call to' note.
  return [&]<int I>() { // expected-note {{in call to}}
    [[assume(
      xs...[I] == 2
    )]];
    [[assume(
      xs...[I + 1] == 0 // expected-note {{assumption evaluated to false}}
    )]];
    return xs...[I];
  }.template operator()<1>();
}

static_assert(test(1, 2, 3, 5, 6) == 2); // expected-error {{not an integral constant expression}} \
                                         // expected-note {{in call to}}

#endif

} // namespace GH114787
