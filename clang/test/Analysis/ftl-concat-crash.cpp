// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

typedef unsigned long size_t;

namespace android {
namespace ftl {

template <typename T>
struct StaticString {
  static constexpr size_t N = 1;
  char view[2];
  constexpr StaticString(T) : view{'a', '\0'} {}
};

template <size_t N, typename... Ts>
struct Concat;

template <size_t N, typename T, typename... Ts>
struct Concat<N, T, Ts...> : Concat<N + StaticString<T>::N, Ts...> {
  explicit constexpr Concat(T v, Ts... args) {
    append(v, args...);
  }

protected:
  constexpr Concat() = default;

  constexpr void append(T v, Ts... args) {
    StaticString<T> str(v);
    this->buffer[this->pos] = str.view[0];
    this->pos++;

    using Base = Concat<N + StaticString<T>::N, Ts...>;
    this->Base::append(args...);
  }
};

template <size_t N>
struct Concat<N> {
protected:
  constexpr Concat() : pos(0) {}
  constexpr void append() {
    buffer[pos] = '\0';
  }

  char buffer[N + 1];
  size_t pos;
};

template <typename... Ts>
Concat(Ts&&...) -> Concat<0, Ts...>;

} // namespace ftl
} // namespace android

void reproduce(unsigned long iteration) {
  android::ftl::Concat trace("TimerIteration #", iteration);
  (void)trace;
}

void test_deep_recursion() {
  reproduce(1);
}
