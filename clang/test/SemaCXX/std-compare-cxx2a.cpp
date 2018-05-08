// Test diagnostics for ill-formed STL <compare> headers.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -fsyntax-only -pedantic -verify -Wsign-compare -std=c++2a %s

void compare_not_found_test() {
  // expected-error@+1 {{cannot deduce return type of 'operator<=>' because type 'std::partial_ordering' was not found; include <compare>}}
  (void)(0.0 <=> 42.123);
}

namespace std {
inline namespace __1 {
struct partial_ordering; // expected-note {{forward declaration}}
}
} // namespace std

auto compare_incomplete_test() {
  // expected-error@+1 {{incomplete type 'std::partial_ordering' where a complete type is required}}
  return (-1.2 <=> 123.0);
}

namespace std {
inline namespace __1 {
struct partial_ordering {
  unsigned value;
};
} // namespace __1
} // namespace std

auto missing_member_test() {
  // expected-error@+1 {{standard library implementation of 'std::partial_ordering' is not supported; member 'equivalent' is missing}}
  return (1.0 <=> 1.0);
}

namespace std {
inline namespace __1 {
struct strong_ordering {
  long long value;
  static const strong_ordering equivalent; // expected-note {{declared here}}
};
} // namespace __1
} // namespace std

auto test_non_constexpr_var() {
  // expected-error@+1 {{standard library implementation of 'std::strong_ordering' is not supported; member 'equivalent' does not have expected form}}
  return (1 <=> 0);
}

namespace std {
inline namespace __1 {
struct strong_equality {
  char value = 0;
  constexpr strong_equality() = default;
  // non-trivial
  constexpr strong_equality(strong_equality const &other) : value(other.value) {}
};
} // namespace __1
} // namespace std

struct Class {};
using MemPtr = void (Class::*)(int);

auto test_non_trivial(MemPtr LHS, MemPtr RHS) {
  // expected-error@+1 {{standard library implementation of 'std::strong_equality' is not supported; the type is not trivially copyable}}
  return LHS <=> RHS;
}
