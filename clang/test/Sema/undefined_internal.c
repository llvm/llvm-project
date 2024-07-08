// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic-errors

static void f(void); // expected-error {{function 'f' has internal linkage but is not defined}}

int main(void)
{
    f;
    // expected-note@-1 {{used here}}
    // expected-warning@-2 {{expression result unused}}
}

