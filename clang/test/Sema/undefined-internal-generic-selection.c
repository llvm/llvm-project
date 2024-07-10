// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic-errors

// expected-no-diagnostics

static int f(void);

int main(void)
{
    int x = _Generic(f(), int: 10, short: 20);
}
