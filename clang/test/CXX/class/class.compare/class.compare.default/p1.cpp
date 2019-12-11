// RUN: %clang_cc1 -std=c++2a -verify %s

struct B {};
bool operator==(const B&, const B&) = default; // expected-error {{equality comparison operator can only be defaulted in a class definition}} expected-note {{candidate}}
bool operator<=>(const B&, const B&) = default; // expected-error {{three-way comparison operator can only be defaulted in a class definition}}

template<typename T = void>
  bool operator<(const B&, const B&) = default; // expected-error {{comparison operator template cannot be defaulted}}

struct A {
  friend bool operator==(const A&, const A&) = default;
  friend bool operator!=(const A&, const B&) = default; // expected-error {{invalid parameter type for defaulted equality comparison operator; found 'const B &', expected 'A' or 'const A &'}}
  friend bool operator!=(const B&, const B&) = default; // expected-error {{invalid parameter type for defaulted equality comparison}}
  friend bool operator<(const A&, const A&);
  friend bool operator<(const B&, const B&) = default; // expected-error {{invalid parameter type for defaulted relational comparison}}
  friend bool operator>(A, A) = default; // expected-warning {{implicitly deleted}}

  bool operator<(const A&) const;
  bool operator<=(const A&) const = default;
  bool operator==(const A&) const volatile && = default; // surprisingly, OK
  bool operator<=>(const A&) = default; // expected-error {{defaulted member three-way comparison operator must be const-qualified}}
  bool operator>=(const B&) const = default; // expected-error-re {{invalid parameter type for defaulted relational comparison operator; found 'const B &', expected 'const A &'{{$}}}}
  static bool operator>(const B&) = default; // expected-error {{overloaded 'operator>' cannot be a static member function}}
  friend bool operator>(A, const A&) = default; // expected-error {{must have the same type}} expected-note {{would be the best match}}

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


namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering equal, greater, less;
  };
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
  constexpr strong_ordering strong_ordering::less = {-1};
}

namespace LookupContext {
  struct A {};

  namespace N {
    template <typename T> auto f() {
      bool operator==(const T &, const T &);
      bool operator<(const T &, const T &);
      struct B {
        T a;
        std::strong_ordering operator<=>(const B &) const = default;
      };
      return B();
    }

    auto g() {
      struct Cmp { Cmp(std::strong_ordering); };
      Cmp operator<=>(const A&, const A&);
      bool operator!=(const Cmp&, int);
      struct B {
        A a;
        Cmp operator<=>(const B &) const = default;
      };
      return B();
    }

    auto h() {
      struct B;
      bool operator==(const B&, const B&);
      bool operator!=(const B&, const B&); // expected-note 2{{best match}}
      std::strong_ordering operator<=>(const B&, const B&);
      bool operator<(const B&, const B&); // expected-note 2{{best match}}
      bool operator<=(const B&, const B&); // expected-note 2{{best match}}
      bool operator>(const B&, const B&); // expected-note 2{{best match}}
      bool operator>=(const B&, const B&); // expected-note 2{{best match}}

      struct B {
        bool operator!=(const B&) const = default; // expected-warning {{implicitly deleted}} expected-note {{deleted here}}
        bool operator<(const B&) const = default; // expected-warning {{implicitly deleted}} expected-note {{deleted here}}
        bool operator<=(const B&) const = default; // expected-warning {{implicitly deleted}} expected-note {{deleted here}}
        bool operator>(const B&) const = default; // expected-warning {{implicitly deleted}} expected-note {{deleted here}}
        bool operator>=(const B&) const = default; // expected-warning {{implicitly deleted}} expected-note {{deleted here}}
      };
      return B();
    }
  }

  namespace M {
    bool operator==(const A &, const A &) = delete;
    bool operator<(const A &, const A &) = delete;
    bool cmp = N::f<A>() < N::f<A>();

    void operator<=>(const A &, const A &) = delete;
    auto cmp2 = N::g() <=> N::g();

    void use_h() {
      N::h() != N::h(); // expected-error {{implicitly deleted}}
      N::h() < N::h(); // expected-error {{implicitly deleted}}
      N::h() <= N::h(); // expected-error {{implicitly deleted}}
      N::h() > N::h(); // expected-error {{implicitly deleted}}
      N::h() >= N::h(); // expected-error {{implicitly deleted}}
    }
  }
}

namespace P1946 {
  struct A {
    friend bool operator==(A &, A &); // expected-note {{would lose const qualifier}}
  };
  struct B {
    A a; // expected-note {{no viable comparison}}
    friend bool operator==(B, B) = default; // ok
    friend bool operator==(const B&, const B&) = default; // expected-warning {{deleted}}
  };
}
