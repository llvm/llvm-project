// RUN: %clang_cc1 -std=c++23 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++20 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++17 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++14 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++11 %s -verify -Wno-unused "-DTYPE_CAST="

// RUN: %clang_cc1 -std=c++23 %s -verify "-DTYPE_CAST=(void)"
// RUN: %clang_cc1 -std=c++20 %s -verify "-DTYPE_CAST=(void)"
// RUN: %clang_cc1 -std=c++17 %s -verify "-DTYPE_CAST=(void)"
// RUN: %clang_cc1 -std=c++14 %s -verify "-DTYPE_CAST=(void)"
// RUN: %clang_cc1 -std=c++11 %s -verify "-DTYPE_CAST=(void)"

void test() {

TYPE_CAST [] constexpr { return true; };
#if __cplusplus <= 201402L
// expected-warning@-2 {{is a C++17 extension}}
#endif
#if __cplusplus <= 202002L
// expected-warning@-5 {{lambda without a parameter clause is a C++23 extension}}
#endif
TYPE_CAST []() mutable //
    mutable             // expected-error{{cannot appear multiple times}}
    mutable {};         // expected-error{{cannot appear multiple times}}

#if __cplusplus > 201402L
TYPE_CAST [] () constexpr mutable constexpr { }; //expected-error{{cannot appear multiple times}}
TYPE_CAST []() mutable constexpr { };
TYPE_CAST []() constexpr { };
TYPE_CAST []() constexpr mutable { };
TYPE_CAST [] () constexpr
                  mutable
                  constexpr   //expected-error{{cannot appear multiple times}}
                  mutable     //expected-error{{cannot appear multiple times}}
                  mutable     //expected-error{{cannot appear multiple times}}
                  constexpr   //expected-error{{cannot appear multiple times}}
                  constexpr   //expected-error{{cannot appear multiple times}}
                  { };

#else
auto L = []() mutable constexpr {return 0; }; //expected-warning{{is a C++17 extension}}
TYPE_CAST []() constexpr { return 0;};//expected-warning{{is a C++17 extension}}
TYPE_CAST []() constexpr mutable { return 0; }; //expected-warning{{is a C++17 extension}}
#endif

}


