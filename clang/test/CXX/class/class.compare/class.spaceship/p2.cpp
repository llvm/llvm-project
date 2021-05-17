// RUN: %clang_cc1 -std=c++2a -verify %s

namespace std {
  class strong_ordering {
    int n;
    constexpr strong_ordering(int n) : n(n) {}
  public:
    static const strong_ordering less, equal, greater;
    bool operator!=(int) { return n != 0; }
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};

  class weak_ordering {
    int n;
    constexpr weak_ordering(int n) : n(n) {}
  public:
    constexpr weak_ordering(strong_ordering o);
    static const weak_ordering less, equivalent, greater;
    bool operator!=(int) { return n != 0; }
  };
  constexpr weak_ordering weak_ordering::less{-1},
      weak_ordering::equivalent{0}, weak_ordering::greater{1};

  class partial_ordering {
    int n;
    constexpr partial_ordering(int n) : n(n) {}
  public:
    constexpr partial_ordering(strong_ordering o);
    constexpr partial_ordering(weak_ordering o);
    static const partial_ordering less, equivalent, greater, unordered;
    bool operator!=(int) { return n != 0; }
  };
  constexpr partial_ordering partial_ordering::less{-1},
      partial_ordering::equivalent{0}, partial_ordering::greater{1},
      partial_ordering::unordered{2};
}

namespace DeducedNotCat {
  struct A {
    A operator<=>(const A&) const; // expected-note {{selected 'operator<=>' for member 'a' declared here}}
  };
  struct B {
    A a; // expected-note {{return type 'DeducedNotCat::A' of three-way comparison for member 'a' is not a standard comparison category type}}
    auto operator<=>(const B&) const = default; // expected-warning {{implicitly deleted}}
  };
}

namespace DeducedVsSynthesized {
  struct A {
    bool operator==(const A&) const;
    bool operator<(const A&) const;
  };
  struct B {
    A a; // expected-note {{no viable three-way comparison function for member 'a'}}
    auto operator<=>(const B&) const = default; // expected-warning {{implicitly deleted}}
  };
}

namespace Deduction {
  template<typename T> struct wrap {
    T t;
    friend auto operator<=>(const wrap&, const wrap&) = default;
  };

  using strong = wrap<int>;
  using strong2 = wrap<int*>;
  struct weak {
    friend std::weak_ordering operator<=>(weak, weak);
  };
  using partial = wrap<float>;

  template<typename ...T> struct A : T... {
    friend auto operator<=>(const A&, const A&) = default;
  };

  template<typename Expected, typename ...Ts> void f() {
    using T = Expected; // expected-note {{previous}}
    using T = decltype(A<Ts...>() <=> A<Ts...>()); // expected-error {{different type}}
    void(A<Ts...>() <=> A<Ts...>()); // trigger synthesis of body
  }

  template void f<std::strong_ordering>();
  template void f<std::strong_ordering, strong>();
  template void f<std::strong_ordering, strong, strong2>();

  template void f<std::weak_ordering, weak>();
  template void f<std::weak_ordering, weak, strong>();
  template void f<std::weak_ordering, strong, weak>();

  template void f<std::partial_ordering, partial>();
  template void f<std::partial_ordering, weak, partial>();
  template void f<std::partial_ordering, strong, partial>();
  template void f<std::partial_ordering, partial, weak>();
  template void f<std::partial_ordering, partial, strong>();
  template void f<std::partial_ordering, weak, partial, strong>();

  // Check that the above mechanism works.
  template void f<std::strong_ordering, weak>(); // expected-note {{instantiation of}}

  std::strong_ordering x = A<strong>() <=> A<strong>();
}

namespace PR44723 {
  // Make sure we trigger return type deduction for a callee 'operator<=>'
  // before inspecting its return type.
  template<int> struct a {
    friend constexpr auto operator<=>(a const &lhs, a const &rhs) {
      return std::strong_ordering::equal;
    }
  };
  struct b {
    friend constexpr auto operator<=>(b const &, b const &) = default;
    a<0> m_value;
  };
  std::strong_ordering cmp_b = b() <=> b();

  struct c {
    auto operator<=>(const c&) const&; // expected-note {{selected 'operator<=>' for base class 'c' declared here}}
  };
  struct d : c { // expected-note {{base class 'c' declared here}}
    friend auto operator<=>(const d&, const d&) = default; // #d
    // expected-error@#d {{return type of defaulted 'operator<=>' cannot be deduced because three-way comparison for base class 'c' has a deduced return type and is not yet defined}}
    // expected-warning@#d {{implicitly deleted}}
  };
  auto c::operator<=>(const c&) const& { // #c
    return std::strong_ordering::equal;
  }
  // expected-error@+1 {{overload resolution selected deleted operator '<=>'}}
  std::strong_ordering cmp_d = d() <=> d();
  // expected-note@#c 2{{candidate}}
  // expected-note@#d {{candidate function has been implicitly deleted}}
}

namespace BadDeducedType {
  struct A {
    // expected-error@+1 {{deduced return type for defaulted three-way comparison operator must be 'auto', not 'auto &'}}
    friend auto &operator<=>(const A&, const A&) = default;
  };

  struct B {
    // expected-error@+1 {{deduced return type for defaulted three-way comparison operator must be 'auto', not 'const auto'}}
    friend const auto operator<=>(const B&, const B&) = default;
  };

  template<typename T> struct X {}; // expected-note {{here}}
  struct C {
    // expected-error@+1 {{deduction not allowed in function return type}}
    friend X operator<=>(const C&, const C&) = default;
  };

  template<typename T> concept CmpCat = true;
  struct D {
    // expected-error@+1 {{deduced return type for defaulted three-way comparison operator must be 'auto', not 'CmpCat auto'}}
    friend CmpCat auto operator<=>(const D&, const D&) = default;
  };
}

namespace PR48856 {
  struct A {
    auto operator<=>(const A &) const = default; // expected-warning {{implicitly deleted}}
    void (*x)();                                 // expected-note {{because there is no viable three-way comparison function for member 'x'}}
  };

  struct B {
    auto operator<=>(const B &) const = default; // expected-warning {{implicitly deleted}}
    void (B::*x)();                              // expected-note {{because there is no viable three-way comparison function for member 'x'}}
  };

  struct C {
    auto operator<=>(const C &) const = default; // expected-warning {{implicitly deleted}}
    int C::*x;                                   // expected-note {{because there is no viable three-way comparison function for member 'x'}}
  };
}
