// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c11 -Wno-pointer-arith -Wno-gnu-alignof-expression -Wno-unused -pedantic-errors

static void *a(void); // expected-error {{function 'a' has internal linkage but is not defined}}
static void *b(void); // expected-error {{function 'b' has internal linkage but is not defined}}
static void *c(void); // expected-error {{function 'c' has internal linkage but is not defined}}
static void *d(void); // expected-error {{function 'd' has internal linkage but is not defined}}
static void *no_err(void);

int main(void)
{
    a; // expected-note {{used here}}

    int i = _Alignof(no_err);

    int j = _Generic(&no_err, void *(*)(void): 0);

    void *k = _Generic(&no_err, void *(*)(void): b(), default: 0); // expected-note {{used here}}

    // FIXME according to the C standard there should be no error if the undefined internal is
    // "part of the expression in a generic association that is not the result expression of its generic selection;"
    // but, currently, clang wrongly emits an error in this case
    k = _Generic(&no_err, void *(*)(void): 0, default: c()); // expected-note {{used here}}

    k = _Generic(&no_err, int (*)(void) : 0, default : d()); // expected-note {{used here}}

    int l = sizeof(no_err);

    __typeof__(&no_err) x;
}
