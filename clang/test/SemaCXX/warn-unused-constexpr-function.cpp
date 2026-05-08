// RUN: %clang_cc1 -std=c++14 -Wunused-function -fsyntax-only %s

static constexpr bool returnInt(int) { return true; }

template <bool B>
struct select;

template <>
struct select<true> {
  using type = int;
};

template <>
struct select<false> {
  using type = float;
};

template <typename T>
typename select<returnInt(T{})>::type make() {
  return T{};
}

int makeInt() { return make<int>(); }
float makeFloat() { return make<float>(); }
