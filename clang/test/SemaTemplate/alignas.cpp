// RUN: %clang_cc1 -std=c++11 -verify %s

// expected-no-diagnostics
using size_t = decltype(sizeof(0));

template<typename T, typename U>
constexpr T max(T t, U u) { return t > u ? t : u; }

template<typename T, typename ...Ts>
constexpr auto max(T t, Ts ...ts) -> decltype(max(t, max(ts...))) {
  return max(t, max(ts...));
}

template<typename...T> struct my_union {
  alignas(T...) char buffer[max(sizeof(T)...)];
};

struct alignas(8) A { char c; };
struct alignas(4) B { short s; };
struct C { char a[16]; };

static_assert(sizeof(my_union<A, B, C>) == 16, "");
static_assert(alignof(my_union<A, B, C>) == 8, "");

namespace PR35028 {
  template<class X, int Alignment> struct alignas(X) alignas(long long) alignas(long double) alignas(Alignment) Aligned {
    union {
      long long align1;
      long double align2;
      char data[sizeof(X)];
    };
  };
  Aligned<int, 1> a;
}
