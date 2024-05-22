// RUN: %clang_cc1 %s -verify -Wconversion

// Don't crash
#define A(desc) _Pragma("clang diagnostic push")  _Pragma("clang diagnostic ignored \"-Wparentheses\"") _Pragma("clang diagnostic pop")
#define B(desc) A(desc)
B(_Pragma("clang diagnostic ignored \"-Wparentheses\""))


#define EMPTY(x)
#define INACTIVE(x) EMPTY(x)

#define ID(x) x
#define ACTIVE(x) ID(x)

// This should be ignored..
INACTIVE(_Pragma("clang diagnostic ignored \"-Wconversion\""))

#define IGNORE_CONV _Pragma("clang diagnostic ignored \"-Wconversion\"") _Pragma("clang diagnostic ignored \"-Wconversion\"")

// ..as should this.
INACTIVE(IGNORE_CONV)

#define IGNORE_POPPUSH(Pop, Push, W, D) Push W D Pop
IGNORE_POPPUSH(_Pragma("clang diagnostic pop"), _Pragma("clang diagnostic push"),
               _Pragma("clang diagnostic ignored \"-Wconversion\""), int q = (double)1.0);

int x1 = (double)1.0; // expected-warning {{implicit conversion}}

ACTIVE(_Pragma) ("clang diagnostic ignored \"-Wconversion\"")) // expected-error {{_Pragma takes a parenthesized string literal}} \
                                      expected-error {{expected identifier or '('}} expected-error {{expected ')'}} expected-note {{to match this '('}}

// This should disable the warning.
ACTIVE(IGNORE_CONV)

int x2 = (double)1.0;
