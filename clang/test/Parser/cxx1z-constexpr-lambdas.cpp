// RUN: %clang_cc1 -std=c++23 %s -verify
// RUN: %clang_cc1 -std=c++20 %s -verify
// RUN: %clang_cc1 -std=c++17 %s -verify
// RUN: %clang_cc1 -std=c++14 %s -verify
// RUN: %clang_cc1 -std=c++11 %s -verify

auto XL0 = [] constexpr { return true; };
#if __cplusplus <= 201402L
// expected-warning@-2 {{is a C++17 extension}}
#endif
#if __cplusplus <= 202002L
// expected-warning@-5 {{lambda without a parameter clause is a C++23 extension}}
#endif
auto XL1 = []() mutable //
    mutable             // expected-error{{cannot appear multiple times}}
    mutable {};         // expected-error{{cannot appear multiple times}}

#if __cplusplus > 201402L
auto XL2 = [] () constexpr mutable constexpr { }; //expected-error{{cannot appear multiple times}}
auto L = []() mutable constexpr { };
auto L2 = []() constexpr { };
auto L4 = []() constexpr mutable { }; 
auto XL16 = [] () constexpr
                  mutable
                  constexpr   //expected-error{{cannot appear multiple times}}
                  mutable     //expected-error{{cannot appear multiple times}}
                  mutable     //expected-error{{cannot appear multiple times}}
                  constexpr   //expected-error{{cannot appear multiple times}}
                  constexpr   //expected-error{{cannot appear multiple times}}
                  { };

#else
auto L = []() mutable constexpr {return 0; }; //expected-warning{{is a C++17 extension}}
auto L2 = []() constexpr { return 0;};//expected-warning{{is a C++17 extension}}
auto L4 = []() constexpr mutable { return 0; }; //expected-warning{{is a C++17 extension}}
#endif


