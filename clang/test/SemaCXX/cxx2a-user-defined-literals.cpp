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
#endif
