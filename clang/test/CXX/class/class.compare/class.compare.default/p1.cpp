// RUN: %clang_cc1 -std=c++2a -verify %s

struct B {};

template<typename T = void>
  bool operator<(const B&, const B&) = default; // expected-error {{comparison operator template cannot be defaulted}}

struct A {
  friend bool operator==(const A&, const A&) = default;
  friend bool operator!=(const A&, const B&) = default; // expected-error {{parameters for defaulted equality comparison operator must have the same type (found 'const A &' vs 'const B &')}}
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

namespace evil1 {
template <class T> struct Bad {
  // expected-error@+1{{found 'const float &'}}
  bool operator==(T const &) const = default;
  Bad(int = 0);
};

template <class T> struct Weird {
  // expected-error@+1{{'float' cannot be used prior to '::'}}
  bool operator==(typename T::Weird_ const &) const = default;
  Weird(int = 0);
};

struct evil {
  using Weird_ = Weird<evil>;
};
template struct Bad<float>;   // expected-note{{evil1::Bad<float>' requested}}
template struct Weird<float>; // expected-note{{evil1::Weird<float>' requested}}
template struct Weird<evil>;

} // namespace evil1

namespace P1946 {
  struct A {
    friend bool operator==(A &, A &); // expected-note {{would lose const qualifier}}
  };
  struct B {
    A a; // expected-note {{no viable three-way comparison}}
    friend bool operator==(B, B) = default; // ok
    friend bool operator==(const B&, const B&) = default; // expected-warning {{deleted}}
  };
}

namespace p2085 {
// out-of-class defaulting

struct S1 {
  bool operator==(S1 const &) const;
};

bool S1::operator==(S1 const &) const = default;

bool F1(S1 &s) {
  return s != s;
}

struct S2 {
  friend bool operator==(S2 const &, S2 const &);
};

bool operator==(S2 const &, S2 const &) = default;
bool F2(S2 &s) {
  return s != s;
}

struct S3 {};                                      // expected-note{{here}}
bool operator==(S3 const &, S3 const &) = default; // expected-error{{not a friend}}

struct S4;                                         // expected-note{{forward declaration}}
bool operator==(S4 const &, S4 const &) = default; // expected-error{{not a friend}}

struct S5;                         // expected-note 3{{forward declaration}}
bool operator==(S5, S5) = default; // expected-error{{not a friend}} expected-error 2{{has incomplete type}}

enum e {};
bool operator==(e, int) = default; // expected-error{{expected class or reference to a constant class}}

bool operator==(e *, int *) = default; // expected-error{{must have at least one}}
} // namespace p2085

namespace p2085_2 {
template <class T> struct S6 {
  // expected-error@+2{{found 'const int &'}}
  // expected-error@+1{{found 'const float &'}}
  bool operator==(T const &) const;
};
template <class T> bool S6<T>::operator==(T const &) const = default;

template struct S6<int>; // expected-note{{S6<int>::operator==' requested}}

void f1() {
  S6<float> a;
  (void)(a == 0); // expected-note{{S6<float>::operator==' requested}}
}

template <class T> struct S7 {
  // expected-error@+2{{'float' cannot be used}}
  // expected-error@+1{{'int' cannot be used}}
  bool operator==(typename T::S7_ const &) const;
  S7(int = 0);
};
template <class T> bool S7<T>::operator==(typename T::S7_ const &) const = default;

struct evil {
  using S7_ = S7<evil>;
};
template struct S7<float>; // expected-note{{S7<float>' requested}}

void f2() {
  S7<int> a; // expected-note{{S7<int>' requested}}
  S7<evil> b;
  (void)(a == 0); // expected-error{{invalid operands}}
  (void)(b == 0);
}
} // namespace p2085_2
