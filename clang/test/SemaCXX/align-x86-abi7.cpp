// RUN: %clang_cc1 -std=c++11 -triple i386-apple-darwin9 -fsyntax-only -verify -fclang-abi-compat=7 %s
// expected-no-diagnostics

using size_t = decltype(sizeof(0));

template <typename T, size_t Preferred>
struct check_alignment {
  using type = T;
  static type value;

  static_assert(__alignof__(value) == Preferred, "__alignof__(value) != Preferred");
  static_assert(__alignof__(type) == Preferred, "__alignof__(type) != Preferred");
  static_assert(alignof(type) == Preferred, "alignof(type) != Preferred");
};

// PR3433
template struct check_alignment<double, 8>;
template struct check_alignment<long long, 8>;
template struct check_alignment<unsigned long long, 8>;

// PR6362
template struct check_alignment<double[3], 8>;

enum big_enum { x = 18446744073709551615ULL };
template struct check_alignment<big_enum, 8>;
