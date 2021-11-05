// RUN: %clang_cc1 -std=c++20 -verify %s -Wno-tautological-compare

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};
}

template <typename T>
void f(int i, int* pi, T* pt, T t) {
  (void)(i <=> i);
  (void)(i <=> pi); // expected-error {{comparison between pointer and integer}}
  (void)(i <=> pt);
  (void)(pi <=> pt);
  (void)(pi <=> t);
}

