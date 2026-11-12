// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

int a<::> = { 1, 2, 3 };
int b = a<:::a<:0:>:>;
bool c = a<:0:><::b;

template<int &n> void f() {}
template void f<::b>();

#define x a<:: ## : b :>
int d = x; // expected-error {{pasting formed ':::', an invalid preprocessing token}} expected-error {{expected unqualified-id}}

const char xs[] = R"(\
??=\U0000)";
static_assert(sizeof(xs) == 12, "did not revert all changes");
