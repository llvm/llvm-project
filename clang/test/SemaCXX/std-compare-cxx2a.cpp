// Test diagnostics for ill-formed STL <compare> headers.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -fsyntax-only -pedantic -verify -Wsign-compare -std=c++2a %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -fsyntax-only -pedantic -verify -Wsign-compare -std=c++2a -DTEST_TRIVIAL=1 %s

#ifndef TEST_TRIVIAL
void compare_not_found_test() {
  // expected-error@+1 {{cannot use builtin operator '<=>' because type 'std::partial_ordering' was not found; include <compare>}}
  (void)(0.0 <=> 42.123);
}

struct deduction_compare_not_found {
  // expected-error@+1 {{cannot default 'operator<=>' because type 'std::strong_ordering' was not found; include <compare>}}
  friend auto operator<=>(const deduction_compare_not_found&, const deduction_compare_not_found&) = default;
};

struct comparable {
  int operator<=>(comparable);
};
struct default_compare_not_found {
  // expected-error@+1 {{cannot default 'operator<=>' because type 'std::strong_ordering' was not found; include <compare>}}
  friend int operator<=>(const default_compare_not_found&, const default_compare_not_found&) = default;
};
bool b = default_compare_not_found() < default_compare_not_found(); // expected-note {{first required here}}

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
  static const strong_ordering equal; // expected-note {{declared here}}
};
} // namespace __1
} // namespace std

auto test_non_constexpr_var() {
  // expected-error@+1 {{standard library implementation of 'std::strong_ordering' is not supported; member 'equal' does not have expected form}}
  return (1 <=> 0);
}

#else

namespace std {
inline namespace __1 {
struct strong_ordering {
  char value = 0;
  constexpr strong_ordering() = default;
  // non-trivial
  constexpr strong_ordering(strong_ordering const &other) : value(other.value) {}
};
} // namespace __1
} // namespace std

auto test_non_trivial(int LHS, int RHS) {
  // expected-error@+1 {{standard library implementation of 'std::strong_ordering' is not supported; the type is not trivially copyable}}
  return LHS <=> RHS;
}

#endif
