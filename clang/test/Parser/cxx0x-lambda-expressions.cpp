// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected,cxx14ext,cxx17ext,cxx20ext,cxx23ext -std=c++03 -Wno-c99-designator %s -Wno-c++11-extensions
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected,cxx14ext,cxx17ext,cxx20ext,cxx23ext -std=c++11 -Wno-c99-designator %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected,cxx17ext,cxx20ext,cxx23ext          -std=c++14 -Wno-c99-designator %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected,cxx20ext,cxx23ext                   -std=c++17 -Wno-c99-designator %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected,cxx23ext                            -std=c++20 -Wno-c99-designator %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=expected                                     -std=c++23 -Wno-c99-designator %s

enum E { e };

#if __cplusplus >= 201103L
constexpr int id(int n) { return n; }
#endif

class C {

  int f() {
    int foo, bar;

    []; // expected-error {{expected body of lambda expression}}
    [+] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [foo,&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [=,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [] {};
    [=] (int i) {};
    [&] (int) mutable -> void {};
    [foo,bar] () { return 3; };
    [=,&foo] () {};
    [&,foo] () {};
    [this] () {};
    [] () -> class C { return C(); };
    [] () -> enum E { return e; };

    [] -> int { return 0; }; // cxx23ext-warning {{lambda without a parameter clause is a C++23 extension}}
    [] mutable -> int { return 0; }; // cxx23ext-warning {{is a C++23 extension}}

    [](int) -> {}; // PR13652 expected-error {{expected a type}}
    return 1;
  }

  void designator_or_lambda() {
    typedef int T;
    const int b = 0;
    const int c = 1;
    int d;
    int a1[1] = {[b] (T()) {}}; // expected-error{{no viable conversion from '(lambda}}
    int a2[1] = {[b] = 1 };
    int a3[1] = {[b,c] = 1 }; // expected-error{{expected ']'}} expected-note {{to match}}
    int a4[1] = {[&b] = 1 }; // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const int *'}}
    int a5[3] = { []{return 0;}() };
    int a6[1] = {[this] = 1 }; // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'C *'}}
    int a7[1] = {[d(0)] { return d; } ()}; // cxx14ext-warning {{initialized lambda captures are a C++14 extension}}
    int a8[1] = {[d = 0] { return d; } ()}; // cxx14ext-warning {{initialized lambda captures are a C++14 extension}}
#if __cplusplus >= 201103L
    int a10[1] = {[id(0)] { return id; } ()}; // cxx14ext-warning {{initialized lambda captures are a C++14 extension}}
#endif
    int a9[1] = {[d = 0] = 1}; // expected-error{{is not an integral constant expression}}
#if __cplusplus >= 201402L
    // expected-note@-2{{constant expression cannot modify an object that is visible outside that expression}}
#endif
#if __cplusplus >= 201103L
    int a11[1] = {[id(0)] = 1};
#endif
  }

  void delete_lambda(int *p) {
    delete [] p;
    delete [] (int*) { new int }; // ok, compound-literal, not lambda
    delete [] { return new int; } (); // expected-error {{'[]' after delete interpreted as 'delete[]'}}
    delete [&] { return new int; } (); // ok, lambda

    delete []() { return new int; }(); // expected-error{{'[]' after delete interpreted as 'delete[]'}}
    delete [](E Enum) { return new int((int)Enum); }(e); // expected-error{{'[]' after delete interpreted as 'delete[]'}}
#if __cplusplus > 201703L
    delete []<int = 0>() { return new int; }(); // expected-error{{'[]' after delete interpreted as 'delete[]'}}
#endif
  }

  // We support init-captures in C++11 as an extension.
  int z;
  void init_capture() {
    [n(0)] () mutable -> int { return ++n; }; // cxx14ext-warning    {{initialized lambda captures are a C++14 extension}}
    [n{0}] { return; };                       // cxx14ext-warning    {{initialized lambda captures are a C++14 extension}}
    [a([&b = z]{})](){};                      // cxx14ext-warning 2  {{initialized lambda captures are a C++14 extension}}
    [n = 0] { return ++n; };                  // expected-error      {{captured by copy in a non-mutable}}
                                              // cxx14ext-warning@-1 {{initialized lambda captures are a C++14 extension}}
    [n = {0}] { return; };                    // expected-error      {{<initializer_list>}}
                                              // cxx14ext-warning@-1 {{initialized lambda captures are a C++14 extension}}

    int x = 4;
    auto y = [&r = x, x = x + 1]() -> int { // cxx14ext-warning 2 {{initialized lambda captures are a C++14 extension}}
      r += 2;
      return x + 2;
    } ();
  }

  void attributes() {
    [] __attribute__((noreturn)){}; // cxx23ext-warning {{lambda without a parameter clause is a C++23 extension}}

    []() [[]]
      mutable {}; // expected-error {{expected body of lambda expression}}

    []() [[]] {};
    []() [[]] -> void {};
    []() mutable [[]] -> void {};
#if __cplusplus >= 201103L
    []() mutable noexcept [[]] -> void {};
#endif

    // Testing GNU-style attributes on lambdas -- the attribute is specified
    // before the mutable specifier instead of after (unlike C++11).
    []() __attribute__((noreturn)) mutable { while(1); };
    []() mutable
      __attribute__((noreturn)) { while(1); }; // expected-error {{expected body of lambda expression}}

    // Testing support for P2173 on adding attributes to the declaration
    // rather than the type.
    [][[]](){}; // cxx23ext-warning {{an attribute specifier sequence in this position is a C++23 extension}}

    []<typename>[[]](){}; // cxx20ext-warning    {{explicit template parameter list for lambdas is a C++20 extension}}
                          // cxx23ext-warning@-1 {{an attribute specifier sequence in this position is a C++23 extension}}

    [][[]]{}; // cxx23ext-warning {{an attribute specifier sequence in this position is a C++23 extension}}
  }

  void missing_parens() {
    [] mutable {}; // cxx23ext-warning {{is a C++23 extension}}
#if __cplusplus >= 201103L
    [] noexcept {}; // cxx23ext-warning {{is a C++23 extension}}
#endif
  }
};

template <typename>
void PR22122() {
  [](int) -> {}; // expected-error {{expected a type}}
}

template void PR22122<int>();

namespace PR42778 {
struct A {
  template <class F> A(F&&) {}
};

struct S {
  void mf() { A(([*this]{})); } // cxx17ext-warning {{'*this' by copy is a C++17 extension}}
};
}

struct S {
  template <typename T>
  void m (T x =[0); // expected-error{{expected variable name or 'this' in lambda capture list}}
} s;

struct U {
  template <typename T>
  void m_fn1(T x = 0[0); // expected-error{{expected ']'}} expected-note{{to match this '['}}
} *U;
