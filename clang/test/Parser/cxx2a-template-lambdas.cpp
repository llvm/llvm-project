// RUN: %clang_cc1 -std=c++23 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++20 %s -verify -Wno-unused "-DTYPE_CAST="
// RUN: %clang_cc1 -std=c++23 %s -verify "-DTYPE_CAST=(void)"
// RUN: %clang_cc1 -std=c++20 %s -verify "-DTYPE_CAST=(void)"

void test() {

TYPE_CAST []<> { }; //expected-error {{cannot be empty}}

TYPE_CAST []<typename T1, typename T2> { };
TYPE_CAST []<typename T1, typename T2>(T1 arg1, T2 arg2) -> T1 { };
TYPE_CAST []<typename T>(auto arg) { T t; };
TYPE_CAST []<int I>() { };

// http://llvm.org/PR49736
TYPE_CAST []<auto>(){};
TYPE_CAST []<auto>{};
TYPE_CAST []<auto>() noexcept {};
TYPE_CAST []<auto> noexcept {};
#if __cplusplus <= 202002L
// expected-warning@-2 {{lambda without a parameter clause is a C++23 extension}}
#endif
TYPE_CAST []<auto> requires true {};
TYPE_CAST []<auto> requires true(){};
TYPE_CAST []<auto> requires true() noexcept {};
TYPE_CAST []<auto> requires true noexcept {};
#if __cplusplus <= 202002L
// expected-warning@-2 {{is a C++23 extension}}
#endif
TYPE_CAST []<auto>() noexcept requires true {};
TYPE_CAST []<auto> requires true() noexcept requires true {};

TYPE_CAST []<auto> noexcept requires true {};               // expected-error {{expected body of lambda expression}}
TYPE_CAST []<auto> requires true noexcept requires true {}; // expected-error {{expected body}}
#if __cplusplus <= 202002L
// expected-warning@-3 {{is a C++23 extension}}
// expected-warning@-3 {{is a C++23 extension}}
#endif

}

namespace GH64962 {
void f() {
  [] <typename T>(T i) -> int[] // expected-error {{function cannot return array type 'int[]'}}
                                // extension-warning {{explicit template parameter list for lambdas is a C++20 extension}}
    { return 3; } (v); // expected-error {{use of undeclared identifier 'v'}}
}
}
