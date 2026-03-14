// RUN: %clang_cc1 -fsyntax-only -verify %s

#define bool _Bool
int main(int argc, char** argv)
{
    bool signed;  // expected-error {{'bool' cannot be signed or unsigned}} expected-warning {{declaration does not declare anything}}

    return 0;
}
#undef bool

typedef int bool;

int test2(int argc, char** argv)
{
    bool signed; // expected-error {{'type-name' cannot be signed or unsigned}} expected-warning {{declaration does not declare anything}}
    _Bool signed; // expected-error {{'_Bool' cannot be signed or unsigned}} expected-warning {{declaration does not declare anything}}

    return 0;
}

