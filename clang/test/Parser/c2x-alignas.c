// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

_Alignas(int) struct c1; // expected-warning {{'_Alignas' attribute ignored}}
alignas(int) struct c1; // expected-warning {{'alignas' attribute ignored}}


__attribute__(()) [[]] alignas(int) int a; // expected-none TODO: actually this line should be an error
__attribute__(()) alignas(int) [[]] int b; // expected-error {{an attribute list cannot appear here}}
__attribute__(()) alignas(int) int c; // expected-none
[[]] __attribute__(()) alignas(int) int d; // expected-none
alignas(int) [[]] __attribute__(()) int e; // expected-error {{an attribute list cannot appear here}}

struct C1 {
    __attribute__(()) [[]] alignas(int) int a; // expected-error {{an attribute list cannot appear here}}
    __attribute__(()) alignas(int) [[]] int b; // expected-error {{an attribute list cannot appear here}}
    __attribute__(()) alignas(int) int c; // expected-none
    [[]] __attribute__(()) alignas(int) int d; // expected-none
    alignas(int) [[]] __attribute__(()) int e; // expected-error {{an attribute list cannot appear here}}
};

void fn_with_decl() {
    __attribute__(()) [[]] alignas(int) int a; // expected-error {{an attribute list cannot appear here}}
    __attribute__(()) alignas(int) [[]] int b; // expected-error {{an attribute list cannot appear here}}
    __attribute__(()) alignas(int) int c; // expected-none
    [[]] __attribute__(()) alignas(int) int d; // expected-none
    alignas(int) [[]] __attribute__(()) int e; // expected-error {{an attribute list cannot appear here}}
}
