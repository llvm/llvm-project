// RUN: %clang_cc1 -std=c++2a -verify %s

struct Good1 {
  bool operator==(const Good1&) const = default;
  bool operator!=(const Good1&) const = default;
};
struct Good2 {
  friend bool operator==(const Good2&, const Good2&) = default;
  friend bool operator!=(const Good2&, const Good2&) = default;
};

enum Bool : bool {};
struct Bad {
  bool &operator==(const Bad&) const = default; // expected-error {{return type for defaulted equality comparison operator must be 'bool', not 'bool &'}}
  const bool operator!=(const Bad&) const = default; // expected-error {{return type for defaulted equality comparison operator must be 'bool', not 'const bool'}}
  friend Bool operator==(const Bad&, const Bad&) = default; // expected-error {{return type for defaulted equality comparison operator must be 'bool', not 'Bool'}}
  friend int operator!=(const Bad&, const Bad&) = default; // expected-error {{return type for defaulted equality comparison operator must be 'bool', not 'int'}}
};

template<typename T> struct Ugly {
  T operator==(const Ugly&) const = default; // expected-error {{return type}}
  T operator!=(const Ugly&) const = default; // expected-error {{return type}}
  friend T operator==(const Ugly&, const Ugly&) = default; // expected-error {{return type}}
  friend T operator!=(const Ugly&, const Ugly&) = default; // expected-error {{return type}}
};
template struct Ugly<bool>;
template struct Ugly<int>; // expected-note {{in instantiation of}}
