// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sme -std=c++20 %s -verify=cxx20
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sme -std=c++23 %s -verify=cxx23
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sme -std=c++23 -Wpre-c++23-compat %s -verify=precxx23
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sme -std=c++23 -pedantic %s -verify=cxx23

auto L1 = [] constexpr {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++23 extension}}
auto L2 = []() static {};
// cxx20-warning@-1 {{static lambdas are a C++23 extension}}
// precxx23-warning@-2 {{static lambdas are incompatible with C++ standards before C++23}}
auto L3 = [] static {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++23 extension}}
// cxx20-warning@-2 {{static lambdas are a C++23 extension}}
// precxx23-warning@-3 {{static lambdas are incompatible with C++ standards before C++23}}

namespace GH161070 {
void t1() { int a = [] __arm_streaming; }
// precxx23-error@-1 {{'__arm_streaming' cannot be applied to a declaration}}
// precxx23-error@-2 {{expected body of lambda expression}}
// cxx23-error@-3 {{'__arm_streaming' cannot be applied to a declaration}}
// cxx23-error@-4 {{expected body of lambda expression}}
// cxx20-error@-5 {{'__arm_streaming' cannot be applied to a declaration}}
// cxx20-error@-6 {{expected body of lambda expression}}
// cxx20-warning@-7 {{'__arm_streaming' in this position is a C++23 extension}}
// precxx23-warning@-8 {{'__arm_streaming' in this position is incompatible with C++ standards before C++23}}

void t2() { int a = [] [[assume(true)]]; }
// precxx23-error@-1 {{'assume' attribute cannot be applied to a declaration}}
// precxx23-error@-2 {{expected body of lambda expression}}
// cxx23-error@-3 {{'assume' attribute cannot be applied to a declaration}}
// cxx23-error@-4 {{expected body of lambda expression}}
// cxx20-error@-5 {{'assume' attribute cannot be applied to a declaration}}
// cxx20-error@-6 {{expected body of lambda expression}}
// cxx20-warning@-7 {{an attribute specifier sequence in this position is a C++23 extension}}
// precxx23-warning@-8 {{an attribute specifier sequence in this position is incompatible with C++ standards before C++23}}
}
