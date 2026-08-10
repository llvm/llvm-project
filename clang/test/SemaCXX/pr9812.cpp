// RUN: %clang_cc1 -fsyntax-only -verify %s

int test(int, char**)
{
    bool signed; // expected-error {{'bool' cannot be signed or unsigned}} expected-warning {{declaration does not declare anything}}

    return 0;
}

