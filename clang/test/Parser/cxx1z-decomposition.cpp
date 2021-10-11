// RUN: %clang_cc1 -std=c++17 %s -verify -fcxx-exceptions
// RUN: not %clang_cc1 -std=c++17 %s -emit-llvm-only -fcxx-exceptions

struct S { int a, b, c; };

// A simple-declaration can be a decompsition declaration.
namespace SimpleDecl {
  auto [a_x, b_x, c_x] = S();

  void f(S s) {
    auto [a, b, c] = S();
    {
      for (auto [a, b, c] = S();;) {}
      if (auto [a, b, c] = S(); true) {}
      switch (auto [a, b, c] = S(); 0) { case 0:; }
    }
  }
}

// A for-range-declaration can be a decomposition declaration.
namespace ForRangeDecl {
  extern S arr[10];
  void h() {
    for (auto [a, b, c] : arr) {
    }
  }
}

// Other kinds of declaration cannot.
namespace OtherDecl {
  // A parameter-declaration is not a simple-declaration.
  // This parses as an array declaration.
  void f(auto [a, b, c]); // expected-error {{'auto' not allowed in function prototype}} expected-error {{'a'}}

  void g() {
    // A condition is allowed as a Clang extension.
    // See commentary in test/Parser/decomposed-condition.cpp
    for (; auto [a, b, c] = S(); ) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    if (auto [a, b, c] = S()) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    if (int n; auto [a, b, c] = S()) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    switch (auto [a, b, c] = S()) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{statement requires expression of integer type ('S' invalid)}}
    switch (int n; auto [a, b, c] = S()) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{statement requires expression of integer type ('S' invalid)}}
    while (auto [a, b, c] = S()) {} // expected-warning {{ISO C++17 does not permit structured binding declaration in a condition}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}

    // An exception-declaration is not a simple-declaration.
    try {}
    catch (auto [a, b, c]) {} // expected-error {{'auto' not allowed in exception declaration}} expected-error {{'a'}}
  }

  // A member-declaration is not a simple-declaration.
  class A {
    auto [a, b, c] = S(); // expected-error {{not permitted in this context}}
    static auto [a, b, c] = S(); // expected-error {{not permitted in this context}}
  };
}

namespace GoodSpecifiers {
  void f() {
    int n[1];
    const volatile auto &[a] = n;
  }
}

namespace BadSpecifiers {
  typedef int I1[1];
  I1 n;
  struct S { int n; } s;
  void f() {
    // storage-class-specifiers
    static auto &[a] = n; // expected-warning {{declared 'static' is a C++20 extension}}
    thread_local auto &[b] = n; // expected-warning {{declared 'thread_local' is a C++20 extension}}
    extern auto &[c] = n; // expected-error {{cannot be declared 'extern'}} expected-error {{cannot have an initializer}}
    struct S {
      mutable auto &[d] = n; // expected-error {{not permitted in this context}}

      // function-specifiers
      virtual auto &[e] = n; // expected-error {{not permitted in this context}}
      explicit auto &[f] = n; // expected-error {{not permitted in this context}}

      // misc decl-specifiers
      friend auto &[g] = n; // expected-error {{'auto' not allowed}} expected-error {{friends can only be classes or functions}}
    };
    typedef auto &[h] = n; // expected-error {{cannot be declared 'typedef'}}
    constexpr auto &[i] = n; // expected-error {{cannot be declared 'constexpr'}}
  }

  static constexpr inline thread_local auto &[j1] = n; // expected-error {{cannot be declared with 'constexpr inline' specifiers}}
  static thread_local auto &[j2] = n; // expected-warning {{declared with 'static thread_local' specifiers is a C++20 extension}}

  inline auto &[k] = n; // expected-error {{cannot be declared 'inline'}}

  const int K = 5;
  void g() {
    // defining-type-specifiers other than cv-qualifiers and 'auto'
    S [a] = s; // expected-error {{cannot be declared with type 'S'}}
    decltype(auto) [b] = s; // expected-error {{cannot be declared with type 'decltype(auto)'}}
    auto ([c]) = s; // expected-error {{cannot be declared with parentheses}}

    // FIXME: This error is not very good.
    auto [d]() = s; // expected-error {{expected ';'}} expected-error {{expected expression}}
    auto [e][1] = s; // expected-error {{expected ';'}} expected-error {{requires an initializer}}

    // FIXME: This should fire the 'misplaced array declarator' diagnostic.
    int [K] arr = {0}; // expected-error {{expected ';'}} expected-error {{cannot be declared with type 'int'}} expected-error {{decomposition declaration '[K]' requires an initializer}}
    int [5] arr = {0}; // expected-error {{place the brackets after the name}}

    auto *[f] = s; // expected-error {{cannot be declared with type 'auto *'}} expected-error {{incompatible initializer}}
    auto S::*[g] = s; // expected-error {{cannot be declared with type 'auto BadSpecifiers::S::*'}} expected-error {{incompatible initializer}}

    // ref-qualifiers are OK.
    auto &&[ok_1] = S();
    auto &[ok_2] = s;

    // attributes are OK.
    [[]] auto [ok_3] = s;
    alignas(S) auto [ok_4] = s;

    // ... but not after the identifier or declarator.
    // FIXME: These errors are not very good.
    auto [bad_attr_1 [[]]] = s; // expected-error {{attribute list cannot appear here}} expected-error 2{{}}
    auto [bad_attr_2] [[]] = s; // expected-error {{expected ';'}} expected-error {{}}
  }
}

namespace MultiDeclarator {
  struct S { int n; };
  void f(S s) {
    auto [a] = s, [b] = s; // expected-error {{must be the only declaration}}
    auto [c] = s,  d = s; // expected-error {{must be the only declaration}}
    auto  e  = s, [f] = s; // expected-error {{must be the only declaration}}
    auto g = s, h = s, i = s, [j] = s; // expected-error {{must be the only declaration}}
  }
}

namespace Template {
  int n[3];
  // FIXME: There's no actual rule against this...
  template<typename T> auto [a, b, c] = n; // expected-error {{decomposition declaration template not supported}}
}

namespace Init {
  void f() {
    int arr[1];
    struct S { int n; };
    auto &[bad1]; // expected-error {{decomposition declaration '[bad1]' requires an initializer}}
    const auto &[bad2](S{}, S{}); // expected-error {{initializer for variable '[bad2]' with type 'const auto &' contains multiple expressions}}
    const auto &[bad3](); // expected-error {{expected expression}}
    auto &[good1] = arr;
    auto &&[good2] = S{};
    const auto &[good3](S{});
    S [goodish3] = { 4 }; // expected-error {{cannot be declared with type 'S'}}
    S [goodish4] { 4 }; // expected-error {{cannot be declared with type 'S'}}
  }
}
