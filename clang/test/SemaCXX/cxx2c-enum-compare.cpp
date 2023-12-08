// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify -triple %itanium_abi_triple

enum E1 { e };
enum E2 { f };
void test() {
    int b = e <= 3.7; // expected-error {{invalid comparison of enumeration type 'E1' with floating-point type 'double'}}
    int k = f - e; // expected-error {{invalid arithmetic between different enumeration types ('E2' and 'E1')}}
    int x = 1 ? e : f; // expected-error {{invalid conditional expression between different enumeration types ('E1' and 'E2')}}
}
