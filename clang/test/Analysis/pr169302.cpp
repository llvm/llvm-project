// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core -verify %s

// expected-no-diagnostics

template <typename> struct S;

class Sp {
public:
  template <bool> void M() {}
  template <unsigned> struct I {
    static void IM();
  };
};

template <> struct S<Sp> {
  using F = void (Sp::*)();
  template <bool P> static constexpr F SpM = &Sp::template M<P>;
};

template <bool> constexpr S<Sp>::F S<Sp>::SpM;

template <unsigned X> void Sp::I<X>::IM() {
  using Spec = S<Sp>;
  typename Spec::F E = Spec::template SpM<true>;
}
