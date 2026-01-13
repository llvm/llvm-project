// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-unknown-linux-gnu -verify=expected,cxx17,pre2c -fcxx-exceptions
// RUN: %clang_cc1 -std=c++2b %s -triple x86_64-unknown-linux-gnu -verify=expected,cxx2b,pre2c,post2b -fcxx-exceptions
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-unknown-linux-gnu -verify=expected,cxx2c,post2b -fcxx-exceptions
// RUN: not %clang_cc1 -std=c++17 %s -triple x86_64-unknown-linux-gnu -emit-llvm-only -fcxx-exceptions

struct S { int a, b, c; }; // expected-note 2 {{'S::a' declared here}}

// A simple-declaration can be a structured binding declaration.
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

// A for-range-declaration can be a structured binding declaration.
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
  void f(auto [a, b, c]); // cxx17-error {{'auto' not allowed in function prototype}} expected-error 1+{{'a'}}

  void g() {
    // A condition is allowed as a Clang extension.
    // See commentary in test/Parser/decomposed-condition.cpp
    for (; auto [a, b, c] = S(); ) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    if (auto [a, b, c] = S()) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    if (int n; auto [a, b, c] = S()) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}
    switch (auto [a, b, c] = S()) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{statement requires expression of integer type ('S' invalid)}}
    switch (int n; auto [a, b, c] = S()) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{statement requires expression of integer type ('S' invalid)}}
    while (auto [a, b, c] = S()) {} // pre2c-warning {{structured binding declaration in a condition is a C++2c extension}} expected-error {{value of type 'S' is not contextually convertible to 'bool'}}

    // An exception-declaration is not a simple-declaration.
    try {}
    catch (auto [a, b, c]) {} // expected-error {{'auto' not allowed in exception declaration}} expected-error 1+{{'a'}}
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
    const volatile auto &[a] = n; // post2b-warning {{volatile qualifier in structured binding declaration is deprecated}}
  }
}

namespace BadSpecifiers {
  typedef int I1[1];
  I1 n;
  struct S { int n; } s;
  void f() {
    // storage-class-specifiers
    static auto &[a] = n; // cxx17-warning {{declared 'static' is a C++20 extension}}
    thread_local auto &[b] = n; // cxx17-warning {{declared 'thread_local' is a C++20 extension}}
    extern auto &[c] = n; // expected-error {{cannot be declared 'extern'}} expected-error {{declaration of block scope identifier with linkage cannot have an initializer}}
    struct S {
      mutable auto &[d] = n; // expected-error {{not permitted in this context}}

      // function-specifiers
      virtual auto &[e] = n; // expected-error {{not permitted in this context}}
      explicit auto &[f] = n; // expected-error {{not permitted in this context}}

      // misc decl-specifiers
      friend auto &[g] = n; // expected-error {{'auto' not allowed}} expected-error {{friends can only be classes or functions}}
    };
    typedef auto &[h] = n; // expected-error {{cannot be declared 'typedef'}}
    constexpr auto &[i] = n; // pre2c-error {{cannot be declared 'constexpr'}}
  }

  static constexpr inline thread_local auto &[j1] = n;
  // pre2c-error@-1 {{cannot be declared 'constexpr'}} \
  // expected-error@-1 {{cannot be declared 'inline'}} \
  // cxx17-warning@-1 {{declared 'static' is a C++20 extension}} \
  // cxx17-warning@-1 {{declared 'thread_local' is a C++20 extension}}

  static thread_local auto &[j2] = n;
  // cxx17-warning@-1 {{declared 'static' is a C++20 extension}}\
  // cxx17-warning@-1 {{declared 'thread_local' is a C++20 extension}}


  inline auto &[k] = n; // expected-error {{cannot be declared 'inline'}}

  const int K = 5;
  auto ([c]) = s; // expected-error {{structured binding declaration cannot be declared with parentheses}}
  void g() {
    // defining-type-specifiers other than cv-qualifiers and 'auto'
    S [a] = s; // expected-error {{cannot be declared with type 'S'}}
    decltype(auto) [b] = s; // expected-error {{cannot be declared with type 'decltype(auto)'}}
    auto ([c2]) = s; // cxx17-error {{structured binding declaration cannot be declared with parenthese}} \
                     // post2b-error {{use of undeclared identifier 'c2'}} \
                     // post2b-error {{expected body of lambda expression}} \

    // FIXME: This error is not very good.
    auto [d]() = s; // expected-error {{expected ';'}} expected-error {{expected expression}}
    auto [e][1] = s; // expected-error {{expected ';'}} expected-error {{requires an initializer}}

    // FIXME: This should fire the 'misplaced array declarator' diagnostic.
    int [K] arr = {0}; // expected-error {{expected ';'}} expected-error {{cannot be declared with type 'int'}} expected-error {{structured binding declaration '[K]' requires an initializer}}
    int [5] arr = {0}; // expected-error {{place the brackets after the name}}

    auto *[f] = s; // expected-error {{cannot be declared with type 'auto *'}} expected-error {{incompatible initializer}}
    auto S::*[g] = s; // expected-error {{cannot be declared with type 'auto S::*'}} expected-error {{incompatible initializer}}

    // ref-qualifiers are OK.
    auto &&[ok_1] = S();
    auto &[ok_2] = s;

    // attributes are OK.
    [[]] auto [ok_3] = s;
    alignas(S) auto [ok_4] = s;

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
  // Structured binding template is not allowed.
  template<typename T> auto [a, b, c] = n; // expected-error {{structured binding declaration cannot be a template}}
}

namespace Init {
  void f() {
    int arr[1];
    struct S { int n; };
    auto &[bad1]; // expected-error {{structured binding declaration '[bad1]' requires an initializer}}
    const auto &[bad2](S{}, S{}); // expected-error {{initializer for variable '[bad2]' with type 'const auto &' contains multiple expressions}}
    const auto &[bad3](); // expected-error {{expected expression}}
    auto &[good1] = arr;
    auto &&[good2] = S{};
    const auto &[good3](S{});
    S [goodish3] = { 4 }; // expected-error {{cannot be declared with type 'S'}}
    S [goodish4] { 4 }; // expected-error {{cannot be declared with type 'S'}}
  }
}


namespace attributes {

struct S{
    int a;
    int b = 0;
};

void err() {
    auto [[]] = S{0}; // expected-error {{expected unqualified-id}}
    auto [ alignas(42) a, foo ] = S{0}; // expected-error {{an attribute list cannot appear here}}
    auto [ c, [[]] d ] = S{0}; // expected-error {{an attribute list cannot appear here}}
    auto [ e, alignas(42) f ] = S{0}; // expected-error {{an attribute list cannot appear here}}
}

void ok() {
    auto [ a alignas(42) [[]], b alignas(42) [[]]] = S{0}; // expected-error 2{{'alignas' attribute only applies to variables, data members and tag types}} \
                                                           // pre2c-warning  2{{an attribute specifier sequence attached to a structured binding declaration is a C++2c extension}}
    auto [ c [[]] alignas(42), d [[]] alignas(42) [[]]] = S{0}; // expected-error 2{{'alignas' attribute only applies to variables, data members and tag types}} \
                                                                // pre2c-warning  2{{an attribute specifier sequence attached to a structured binding declaration is a C++2c extension}}
}


auto [G1 [[deprecated]], G2 [[deprecated]]] = S{42}; // #deprecated-here
// pre2c-warning@-1 2{{an attribute specifier sequence attached to a structured binding declaration is a C++2c extension}}

int test() {
  return G1 + G2; // expected-warning {{'G1' is deprecated}} expected-note@#deprecated-here {{here}} \
                  // expected-warning {{'G2' is deprecated}} expected-note@#deprecated-here {{here}}
}

void invalid_attributes() {
  // pre2c-warning@+1 {{an attribute specifier sequence attached to a structured binding declaration is a C++2c extension}}
  auto [a alignas(42) // expected-error {{'alignas' attribute only applies to variables, data members and tag types}}
      [[assume(true), // expected-error {{'assume' attribute cannot be applied to a declaration}}
        carries_dependency, // expected-error {{'carries_dependency' attribute only applies to parameters, Objective-C methods, and functions}}
        fallthrough,  // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}
        likely, // expected-error {{'likely' attribute cannot be applied to a declaration}}
        unlikely, // expected-error {{'unlikely' attribute cannot be applied to a declaration}}
        nodiscard,  // expected-warning {{'nodiscard' attribute only applies to Objective-C methods, enums, structs, unions, classes, functions, function pointers, and typedefs}}
        noreturn,  // expected-error {{'noreturn' attribute only applies to functions}}
        no_unique_address]], // expected-error {{'no_unique_address' attribute only applies to non-bit-field non-static data members}}
    b] = S{0};
}

}
