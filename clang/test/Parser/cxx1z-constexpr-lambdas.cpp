// RUN: %clang_cc1 -std=c++17 %s -verify 
// RUN: %clang_cc1 -std=c++14 %s -verify 
// RUN: %clang_cc1 -std=c++11 %s -verify 


auto XL0 = [] constexpr { }; //expected-error{{requires '()'}} expected-error{{expected body}}
auto XL1 = [] () mutable 
                 mutable     //expected-error{{cannot appear multiple times}}
                 mutable { }; //expected-error{{cannot appear multiple times}}

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


