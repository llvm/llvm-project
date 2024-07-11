// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic-errors

static void *f(void); // expected-error {{function 'f' has internal linkage but is not defined}}

int main(void)
{
    int j = _Generic(&f, void *(*)(void): 10, default: 20);
    // expected-no-diagnostic@-1

    void *k = _Generic(&f, void *(*)(void): f(), default: 20);
    // expected-note@-1 {{used here}}
}
