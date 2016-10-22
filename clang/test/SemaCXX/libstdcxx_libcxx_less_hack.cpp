// This is a test for a hack in Clang that works around a problem introduced by
// DR583: it's no longer possible to compare a pointer against nullptr_t, but
// we still want to permit those comparisons within less<> and friends.

// RUN: %clang_cc1 -verify %s -std=c++14

namespace std {
  template<typename T = void> struct less {};
  template<typename T = void> struct less_equal {};
  template<typename T = void> struct greater {};
  template<typename T = void> struct greater_equal {};

  template<> struct less<> {
    template <class T1, class T2>
    auto operator()(T1 &&t, T2 &&u) const noexcept(noexcept(t < u))
        -> decltype(t < u) {
      return t < u;
    }
  };

  template<> struct less_equal<> {
    template <class T1, class T2>
    auto operator()(T1 &&t, T2 &&u) const noexcept(noexcept(t <= u))
        -> decltype(t <= u) {
      return t <= u;
    }
  };

  template<> struct greater<> {
    template <class T1, class T2>
    auto operator()(T1 &&t, T2 &&u) const noexcept(noexcept(t > u))
        -> decltype(t > u) {
      return t > u;
    }
  };

  template<> struct greater_equal<> {
    template <class T1, class T2>
    auto operator()(T1 &&t, T2 &&u) const noexcept(noexcept(t >= u))
        -> decltype(t >= u) {
      return t >= u;
    }
  };

  template<typename = void> struct unrelated;
  template<> struct unrelated<> {
    template <class T1, class T2>
    auto operator()(T1 &&t, T2 &&u) const noexcept(noexcept(t < u)) // expected-note {{substitution failure}}
        -> decltype(t < u) {
      return t < u;
    }
  };
};

void test(int *p) {
  using namespace std;
  less<>()(p, nullptr);
  less<>()(nullptr, p);
  less_equal<>()(p, nullptr);
  less_equal<>()(nullptr, p);
  greater<>()(p, nullptr);
  greater<>()(nullptr, p);
  greater_equal<>()(p, nullptr);
  greater_equal<>()(nullptr, p);

  unrelated<>()(p, nullptr); // expected-error {{no matching function}}
}
