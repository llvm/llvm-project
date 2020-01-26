// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify
// expected-no-diagnostics

template<typename T, typename U=void>
concept C = true;

namespace ns {
  template<typename T, typename U=void>
  concept D = true;
}

void foo(C auto a,
         C<int> auto b,
         ns::D auto c,
         ns::D<int> auto d,
         const C auto e,
         const C<int> auto f,
         const ns::D auto g,
         const ns::D<int> auto h);