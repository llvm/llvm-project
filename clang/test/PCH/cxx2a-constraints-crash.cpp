// RUN: %clang_cc1 -std=c++2a -fallow-pch-with-compiler-errors -emit-pch %s -o %t -verify
// RUN: %clang_cc1 -std=c++2a -fallow-pch-with-compiler-errors -include-pch %t %s -verify

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

namespace GH99036 {

template <typename T>
concept C;
// expected-error@-1 {{expected '='}} \
// expected-note@-1 {{declared here}}

template <C U> void f();
// expected-error@-1 {{a concept definition cannot refer to itself}}

} // namespace GH99036
