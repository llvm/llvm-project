// RUN: %clang_cc1 -std=c++17 -verify=cxx17 -Wc++2a-compat %s
// RUN: %clang_cc1 -std=c++2a -verify=cxx2a -Wc++17-compat %s

namespace disambig {

// Cases that are valid in C++17 and before, ill-formed in C++20, and that we
// should not treat as explicit(bool) as an extension.
struct A { // cxx2a-note +{{}}
  constexpr A() {}
  constexpr operator bool() { return true; }

  constexpr explicit (A)(int); // #1
  // cxx17-warning@#1 {{will be parsed as explicit(bool)}}
  // cxx2a-error@#1 +{{}} cxx2a-note@#1 +{{}}
  // cxx2a-warning@#1 {{incompatible with C++ standards before C++2a}}

  // This is ill-formed (via a DR change), and shouldn't be recognized as a
  // constructor (the function declarator cannot be parenthesized in a
  // constructor declaration). But accepting it as an extension seems
  // reasonable.
  // FIXME: Produce an ExtWarn for this.
  constexpr explicit (A(float)); // #1b
  // cxx17-warning@#1b {{will be parsed as explicit(bool)}}
  // cxx2a-error@#1b +{{}}
  // cxx2a-warning@#1b {{incompatible with C++ standards before C++2a}}

  explicit (operator int)(); // #2
  // cxx17-warning@#2 {{will be parsed as explicit(bool)}}
  // cxx2a-error@#2 +{{}}
  // cxx2a-warning@#2 {{incompatible with C++ standards before C++2a}}

  explicit (A::operator float)(); // #2b
  // cxx17-warning@#2b {{will be parsed as explicit(bool)}}
  // cxx17-error@#2b {{extra qualification on member}}
  // cxx2a-error@#2b +{{}}
  // cxx2a-warning@#2b {{incompatible with C++ standards before C++2a}}
};

constexpr bool operator+(A) { return true; }

constexpr bool C = false;

// Cases that should (ideally) be disambiguated as explicit(bool) in earlier
// language modes as an extension.
struct B {
  // Looks like a constructor, but not the constructor of B.
  explicit (A()) B(); // #3
  // cxx17-warning@#3 {{C++2a extension}}
  // cxx2a-warning@#3 {{incompatible with C++ standards before C++2a}}

  // Looks like a 'constructor' of C. Actually a constructor of B.
  explicit (C)(B)(A); // #4
  // cxx17-warning@#4 {{C++2a extension}}
  // cxx2a-warning@#4 {{incompatible with C++ standards before C++2a}}

  explicit (operator+(A())) operator int(); // #5
  // cxx17-error@#5 {{requires a type specifier}} cxx17-error@#5 {{expected ';'}}
  // cxx17-warning@#5 {{will be parsed as explicit(bool)}}
  // cxx2a-warning@#5 {{incompatible with C++ standards before C++2a}}
};

}
