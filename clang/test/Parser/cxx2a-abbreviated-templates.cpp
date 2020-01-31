// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify
// expected-no-diagnostics

template<typename T, typename U=void>
concept C = true;

namespace ns {
  template<typename T, typename U=void>
  concept D = true;
}

void foo1(C auto a,
          C<int> auto b,
          ns::D auto c,
          ns::D<int> auto d,
          const C auto e,
          const C<int> auto f,
          const ns::D auto g,
          const ns::D<int> auto h);
void foo2(C auto a);
void foo3(C<int> auto b);
void foo4(ns::D auto c);
void foo5(ns::D<int> auto d);
void foo6(const C auto e);
void foo7(const C<int> auto f);
void foo8(const ns::D auto g);
void foo9(const ns::D<int> auto h);

struct S1 { S1(C auto a,
               C<int> auto b,
               ns::D auto c,
               ns::D<int> auto d,
               const C auto e,
               const C<int> auto f,
               const ns::D auto g,
               const ns::D<int> auto h); };
struct S2 { S2(C auto a); };
struct S3 { S3(C<int> auto b); };
struct S4 { S4(ns::D auto c); };
struct S5 { S5(ns::D<int> auto d); };
struct S6 { S6(const C auto e); };
struct S7 { S7(const C<int> auto f); };
struct S8 { S8(const ns::D auto g); };
struct S9 { S9(const ns::D<int> auto h); };