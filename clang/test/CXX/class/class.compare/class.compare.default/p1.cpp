// RUN: %clang_cc1 -std=c++2a -verify %s

struct B {};
bool operator==(const B&, const B&) = default; // expected-error {{equality comparison operator can only be defaulted in a class definition}}
bool operator<=>(const B&, const B&) = default; // expected-error {{three-way comparison operator can only be defaulted in a class definition}}

template<typename T = void>
  bool operator<(const B&, const B&) = default; // expected-error {{comparison operator template cannot be defaulted}}

struct A {
  friend bool operator==(const A&, const A&) = default;
  friend bool operator!=(const A&, const B&) = default; // expected-error {{invalid parameter type for defaulted equality comparison}}
  friend bool operator!=(const B&, const B&) = default; // expected-error {{invalid parameter type for defaulted equality comparison}}
  friend bool operator<(const A&, const A&);
  friend bool operator<(const B&, const B&) = default; // expected-error {{invalid parameter type for defaulted relational comparison}}
  friend bool operator>(A, A) = default; // expected-error {{invalid parameter type for defaulted relational comparison}}

  bool operator<(const A&) const;
  bool operator<=(const A&) const = default;
  bool operator==(const A&) const volatile && = default; // surprisingly, OK
  bool operator<=>(const A&) = default; // expected-error {{defaulted member three-way comparison operator must be const-qualified}}
  bool operator>=(const B&) const = default; // expected-error {{invalid parameter type for defaulted relational comparison}}
  static bool operator>(const B&) = default; // expected-error {{overloaded 'operator>' cannot be a static member function}}

  template<typename T = void>
    friend bool operator==(const A&, const A&) = default; // expected-error {{comparison operator template cannot be defaulted}}
  template<typename T = void>
    bool operator==(const A&) const = default; // expected-error {{comparison operator template cannot be defaulted}}
};

// FIXME: The wording is not clear as to whether these are valid, but the
// intention is that they are not.
bool operator<(const A&, const A&) = default; // expected-error {{relational comparison operator can only be defaulted in a class definition}}
bool A::operator<(const A&) const = default; // expected-error {{can only be defaulted in a class definition}}

template<typename T> struct Dependent {
  using U = typename T::type;
  bool operator==(U) const = default; // expected-error {{found 'Dependent<Bad>::U'}}
  friend bool operator==(U, U) = default; // expected-error {{found 'Dependent<Bad>::U'}}
};

struct Good { using type = const Dependent<Good>&; };
template struct Dependent<Good>;

struct Bad { using type = Dependent<Bad>&; };
template struct Dependent<Bad>; // expected-note {{in instantiation of}}
