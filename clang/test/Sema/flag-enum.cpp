// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

enum Unscoped { U0 = 1, U1 = 8 };
enum class Scoped { S0 = 1, S1 = 8 };
enum [[clang::flag_enum]] UnscopedFlag { D0 = 1, D1 = 8 };
enum class [[clang::flag_enum]] WithOps { D0 = 1, D1 = 8 };
enum class [[clang::flag_enum]] WithoutOps { D0 = 1, D1 = 8 };
// expected-warning@-1 {{operator| is not available for flag-like enumeration type WithoutOps}} \
// expected-warning@-1 {{operator& is not available for flag-like enumeration type WithoutOps}} \
// expected-warning@-1 {{operator^ is not available for flag-like enumeration type WithoutOps}} \
// expected-warning@-1 {{operator~ is not available for flag-like enumeration type WithoutOps}}

WithOps operator|(WithOps L, WithOps R) {
  return static_cast<WithOps>(static_cast<unsigned>(L) | static_cast<unsigned>(R));
}

WithOps operator&(WithOps L, WithOps R) {
  return static_cast<WithOps>(static_cast<unsigned>(L) & static_cast<unsigned>(R));
}

WithOps operator^(WithOps L, WithOps R) {
  return static_cast<WithOps>(static_cast<unsigned>(L) ^ static_cast<unsigned>(R));
}

WithOps operator~(WithOps L) {
  return static_cast<WithOps>(~static_cast<unsigned>(L));
}

namespace test {
enum class [[clang::flag_enum]] Foo { A=1, B=2 };
// expected-warning@-1 {{operator| is ambiguous for flag-like enumeration type Foo}} \
//   expected-note@#candidate1 {{candidate function}} \
//   expected-note@#candidate2 {{candidate function}} \
// expected-warning@-1 {{operator& is deleted for flag-like enumeration type Foo}} \
//   expected-note@#deleted1 {{candidate function has been explicitly deleted}} \
// expected-warning@-1 {{operator^ is deleted for flag-like enumeration type Foo: reason}} \
//   expected-note@#deleted2 {{candidate function has been explicitly deleted}} \
// expected-warning@-1 {{operator~ is not available for flag-like enumeration type Foo}}

constexpr Foo operator|(Foo lhs, Foo rhs) { // #candidate1
  return static_cast<Foo>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

Foo operator&(Foo L, Foo R) = delete; // #deleted1
Foo operator^(Foo L, Foo R) = delete("reason"); // #deleted2
}

constexpr test::Foo operator|(test::Foo lhs, test::Foo rhs) { // #candidate2
  return static_cast<test::Foo>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}


template <class T>
struct Foo {
  enum class [[clang::flag_enum]] Bar : T { A=1, B=2 }; // #dependent-enum
};

template struct Foo<int>;
// expected-warning@#dependent-enum {{operator| is not available for flag-like enumeration type Bar}} \
// expected-warning@#dependent-enum {{operator& is not available for flag-like enumeration type Bar}} \
// expected-warning@#dependent-enum {{operator^ is not available for flag-like enumeration type Bar}} \
// expected-warning@#dependent-enum {{operator~ is not available for flag-like enumeration type Bar}}
