// RUN: %clang_cc1 -std=c++2a -verify %s

namespace std_example {
  namespace std { template<typename T> T &&move(T &); }

  void g(...);

  template <class... Args> void f(Args... args) {
    auto lm = [&, args...] { return g(args...); };
    lm();

    auto lm2 = [... xs = std::move(args)] { return g(xs...); };
    lm2();
  }
}

template<typename ...T> constexpr int f(int k, T ...t) {
  auto a = [...v = t] (bool b) mutable {
    if (!b) {
      ((v += 1), ...);
      return (__SIZE_TYPE__)0;
    }
    return (v * ... * 1) + sizeof...(v);
  };
  for (int i = 0; i != k; ++i)
    a(false);
  return a(true);
}

static_assert(f(1, 2, 3, 4) == 3 * 4 * 5 + 3);
static_assert(f(5) == 1);

auto q = [...x = 0] {}; // expected-error {{does not contain any unexpanded parameter packs}}

template<typename ...T> constexpr int nested(T ...t) {
  return [...a = t] {
    return [a...] {
      return (a + ...);
    }();
  }();
}
static_assert(nested(1, 2, 3) == 6);
