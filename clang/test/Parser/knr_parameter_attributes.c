// RUN: %clang_cc1 -fsyntax-only -W -Wall -Wno-deprecated-non-prototype -Werror -verify %s -std=c99
// expected-no-diagnostics

int f(int i __attribute__((__unused__)))
{
    return 0;
}
int g(i)
    int i __attribute__((__unused__));
{
    return 0;
}
