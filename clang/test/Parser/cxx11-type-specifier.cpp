// RUN: %clang_cc1 -verify %s -std=c++11 -fcxx-exceptions

// Tests for parsing of type-specifier-seq

struct S {
  operator constexpr int(); // expected-error{{type name does not allow constexpr}}
};
enum E { e };

void f() {
  try {
    (void) new constexpr int; // expected-error{{type name does not allow constexpr}}
  } catch (constexpr int) { // expected-error{{type name does not allow constexpr}}
  }

  // These parse as type definitions, not as type references with braced
  // initializers. Sad but true...
  (void) new struct S {}; // expected-error{{'S' can not be defined in a type specifier}}
  (void) new enum E { e }; // expected-error{{'E' can not be defined in a type specifier}}
}

// And for trailing-type-specifier-seq

auto f() -> unknown; // expected-error{{unknown type name 'unknown'}}
