// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c23 -pedantic-errors

// expected-no-diagnostics

static int f(void);

int main(void)
{
    typeof(&f) x;
}
