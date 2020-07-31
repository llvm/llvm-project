// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, typename U>
concept not_same_as = true;

template <int Kind>
struct subrange {
  template <not_same_as<int> R>
  subrange(R) requires(Kind == 0);

  template <not_same_as<int> R>
  subrange(R) requires(Kind != 0);
};

template <typename R>
subrange(R) -> subrange<42>;

int main() {
  int c;
  subrange s(c);
}

#endif
