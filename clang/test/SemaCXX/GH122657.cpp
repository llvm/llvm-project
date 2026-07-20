// RUN: %clang_cc1 -x c++ -std=c++11 -fblocks -fsyntax-only -verify %s
// expected-no-diagnostics

template <unsigned long long n>
struct Sized {
  char data[n];
};

template <typename T>
int baz() {
  static constexpr auto funcSize = sizeof(__func__);
  static constexpr auto functionSize = sizeof(__FUNCTION__);
  static constexpr auto prettySize = sizeof(__PRETTY_FUNCTION__);

  auto lfunc = []() noexcept(sizeof(__func__) == funcSize) -> Sized<sizeof(__func__)> { return {}; };
  auto lfunction = []() noexcept(sizeof(__FUNCTION__) == functionSize) -> Sized<sizeof(__FUNCTION__)> { return {}; };
  auto lpretty = []() noexcept(sizeof(__PRETTY_FUNCTION__) == prettySize) -> Sized<sizeof(__PRETTY_FUNCTION__)> { return {}; };

  static_assert(sizeof(lfunc()) == 5, "baz");
  static_assert(noexcept(lfunc()) == true, "noexcept");

  static_assert(sizeof(lfunction()) == 5, "baz");
  static_assert(noexcept(lfunction()) == true, "noexcept");

  static_assert(sizeof(lpretty()) == 33, "int baz() [T = int]_block_invoke");
  static_assert(noexcept(lpretty()) == true, "noexcept");

  return 0;
}

int main() {
  static constexpr auto funcSize = sizeof(__func__);
  static constexpr auto functionSize = sizeof(__FUNCTION__);
  static constexpr auto prettySize = sizeof(__PRETTY_FUNCTION__);

  auto lfunc = []() noexcept(sizeof(__func__) == funcSize) -> Sized<sizeof(__func__)> { return {}; };
  auto lfunction = []() noexcept(sizeof(__FUNCTION__) == functionSize) -> Sized<sizeof(__FUNCTION__)> { return {}; };
  auto lpretty = []() noexcept(sizeof(__PRETTY_FUNCTION__) == prettySize) -> Sized<sizeof(__PRETTY_FUNCTION__)> { return {}; };

  static_assert(sizeof(lfunc()) == 5, "main");
  static_assert(noexcept(lfunc()) == true, "noexcept");

  static_assert(sizeof(lfunction()) == 5, "main");
  static_assert(noexcept(lfunction()) == true, "noexcept");

  static_assert(sizeof(lpretty()) == 11, "int main()");
  static_assert(noexcept(lpretty()) == true, "noexcept");
}

