// RUN: %clang_cc1 -std=c++2a -verify %s

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};
}

namespace Deletedness {
  struct A {
    std::strong_ordering operator<=>(const A&) const;
  };
  struct B {
    bool operator==(const B&) const;
    bool operator<(const B&) const;
  };
  struct C {
    std::strong_ordering operator<=>(const C&) const = delete; // expected-note {{deleted}}
  };
  struct D1 {
    bool operator==(const D1&) const;
    std::strong_ordering operator<=>(int) const; // expected-note {{function not viable}} expected-note {{function (with reversed parameter order) not viable}}
    bool operator<(int) const; // expected-note {{function not viable}}
  };
  struct D2 {
    bool operator<(const D2&) const;
    std::strong_ordering operator<=>(int) const; // expected-note {{function not viable}} expected-note {{function (with reversed parameter order) not viable}}
    bool operator==(int) const; // expected-note {{function not viable}}
  };
  struct E {
    bool operator==(const E&) const;
    bool operator<(const E&) const = delete; // expected-note {{deleted}}
  };
  struct F {
    std::strong_ordering operator<=>(const F&) const; // expected-note {{candidate}}
    std::strong_ordering operator<=>(F) const; // expected-note {{candidate}}
  };
  struct G1 {
    bool operator==(const G1&) const;
    void operator<(const G1&) const;
  };
  struct G2 {
    void operator==(const G2&) const;
    bool operator<(const G2&) const;
  };
  struct H {
    void operator<=>(const H&) const;
  };

  // expected-note@#base {{deleted comparison function for base class 'C'}}
  // expected-note@#base {{no viable comparison function for base class 'D1'}}
  // expected-note@#base {{three-way comparison cannot be synthesized because there is no viable function for '<' comparison}}
  // expected-note@#base {{no viable comparison function for base class 'D2'}}
  // expected-note@#base {{three-way comparison cannot be synthesized because there is no viable function for '==' comparison}}
  // expected-note@#base {{deleted comparison function for base class 'E'}}
  // expected-note@#base {{implied comparison for base class 'F' is ambiguous}}
  template<typename T> struct Cmp : T { // #base
    std::strong_ordering operator<=>(const Cmp&) const = default; // expected-note 5{{here}}
  };

  void use(...);
  void f() {
    use(
      Cmp<A>() <=> Cmp<A>(),
      Cmp<B>() <=> Cmp<B>(),
      Cmp<C>() <=> Cmp<C>(), // expected-error {{deleted}}
      Cmp<D1>() <=> Cmp<D1>(), // expected-error {{deleted}}
      Cmp<D2>() <=> Cmp<D2>(), // expected-error {{deleted}}
      Cmp<E>() <=> Cmp<E>(), // expected-error {{deleted}}
      Cmp<F>() <=> Cmp<F>(), // expected-error {{deleted}}
      Cmp<G1>() <=> Cmp<G1>(), // FIXME: ok but synthesized body is ill-formed
      Cmp<G2>() <=> Cmp<G2>(), // FIXME: ok but synthesized body is ill-formed
      Cmp<H>() <=> Cmp<H>(), // FIXME: ok but synthesized body is ill-formed
      0
    );
  }
}
