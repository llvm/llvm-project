// RUN: %clang_cc1 -std=c++2a %s -include %s -verify

#ifndef INCLUDED
#define INCLUDED

#pragma clang system_header
namespace std {
  namespace chrono {
    struct day{};
    struct year{};
  }
  constexpr chrono::day operator"" d(unsigned long long d) noexcept;
  constexpr chrono::year operator"" y(unsigned long long y) noexcept;
}

#else

using namespace std;
chrono::day dec_d = 5d;
chrono::day oct_d = 05d;
chrono::day bin_d = 0b011d;
// expected-error@+3{{no viable conversion from 'int' to 'chrono::day'}}
// expected-note@9{{candidate constructor (the implicit copy constructor)}}
// expected-note@9{{candidate constructor (the implicit move constructor)}}
chrono::day hex_d = 0x44d;
chrono::year y  = 10y;

namespace ignore_class_udl_for_numeric_literals {
  struct A { constexpr A(const char*) {} };
  struct B { constexpr B(char); };
  struct C { constexpr C(int); };
  template<A> void operator""_a();
  template<B> void operator""_b();
  template<C> void operator""_c();
  void test_class_udl_1() {
    1_a; // expected-error {{no matching}}
    1_b; // expected-error {{no matching}}
    1_c; // expected-error {{no matching}}
    "1"_a;
    "1"_b; // expected-error {{no matching}}
    "1"_c; // expected-error {{no matching}}
  }
  template<char...> void operator""_a();
  template<char...> void operator""_b();
  template<char...> void operator""_c();
  void test_class_udl_2() {
    1_a;
    // FIXME: The standard appears to say these two are ambiguous!
    1_b;
    1_c;
    "1"_a;
    "1"_b; // expected-error {{no matching}}
    "1"_c; // expected-error {{no matching}}
  }
}
#endif
