// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-pointer-arith -pedantic-errors

static void f(void); // expected-no-error

int main(void)
{
    int i = _Alignof(f);
    // expected-error@-1 {{'_Alignof' applied to an expression is a GNU extension}}
    // expected-error@-2 {{invalid application of 'alignof' to a function type}}
}
