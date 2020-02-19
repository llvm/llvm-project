// RUN: %clang_cc1 -std=c++17 -verify=cxx17 -Wc++20-compat %s
// RUN: %clang_cc1 -std=c++20 -verify=cxx20 -Wc++17-compat %s

namespace disambig {

// Cases that are valid in C++17 and before, ill-formed in C++20, and that we
// should not treat as explicit(bool) as an extension.
struct A { // cxx20-note +{{}}
  constexpr A() {}
  constexpr operator bool() { return true; }

  constexpr explicit (A)(int); // #1
  // cxx17-warning@#1 {{will be parsed as explicit(bool)}}
  // cxx20-error@#1 +{{}} cxx20-note@#1 +{{}}
  // cxx20-warning@#1 {{incompatible with C++ standards before C++20}}

  // This is ill-formed (via a DR change), and shouldn't be recognized as a
  // constructor (the function declarator cannot be parenthesized in a
  // constructor declaration). But accepting it as an extension seems
  // reasonable.
  // FIXME: Produce an ExtWarn for this.
  constexpr explicit (A(float)); // #1b
  // cxx17-warning@#1b {{will be parsed as explicit(bool)}}
  // cxx20-error@#1b +{{}}
  // cxx20-warning@#1b {{incompatible with C++ standards before C++20}}

  explicit (operator int)(); // #2
  // cxx17-warning@#2 {{will be parsed as explicit(bool)}}
  // cxx20-error@#2 +{{}}
  // cxx20-warning@#2 {{incompatible with C++ standards before C++20}}

  explicit (A::operator float)(); // #2b
  // cxx17-warning@#2b {{will be parsed as explicit(bool)}}
  // cxx17-error@#2b {{extra qualification on member}}
  // cxx20-error@#2b +{{}}
  // cxx20-warning@#2b {{incompatible with C++ standards before C++20}}
};

constexpr bool operator+(A) { return true; }

constexpr bool C = false;

// Cases that should (ideally) be disambiguated as explicit(bool) in earlier
// language modes as an extension.
struct B {
  // Looks like a constructor, but not the constructor of B.
  explicit (A()) B(); // #3
  // cxx17-warning@#3 {{C++20 extension}}
  // cxx20-warning@#3 {{incompatible with C++ standards before C++20}}

  // Looks like a 'constructor' of C. Actually a constructor of B.
  explicit (C)(B)(A); // #4
  // cxx17-warning@#4 {{C++20 extension}}
  // cxx20-warning@#4 {{incompatible with C++ standards before C++20}}

  explicit (operator+(A())) operator int(); // #5
  // cxx17-error@#5 {{requires a type specifier}} cxx17-error@#5 {{expected ';'}}
  // cxx17-warning@#5 {{will be parsed as explicit(bool)}}
  // cxx20-warning@#5 {{incompatible with C++ standards before C++20}}
};

}
