// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify -std=c11 %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -pedantic -verify=pedantic-expected -std=c11 %s
// RUN: %clang_cc1 -verify=ref -std=c11 %s
// RUN: %clang_cc1 -pedantic -verify=pedantic-ref -std=c11 %s

_Static_assert(1, "");
_Static_assert(0 != 1, "");
_Static_assert(1.0 == 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                                // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert(1 && 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                              // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert( (5 > 4) + (3 > 2) == 2, "");
_Static_assert(!!1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                           // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert(!!1, "");

int a = (1 == 1 ? 5 : 3);
_Static_assert(a == 5, ""); // ref-error {{not an integral constant expression}} \
                            // pedantic-ref-error {{not an integral constant expression}} \
                            // expected-error {{not an integral constant expression}} \
                            // pedantic-expected-error {{not an integral constant expression}}


const int b = 3;
_Static_assert(b == 3, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                            // pedantic-expected-warning {{not an integer constant expression}}

/// FIXME: The new interpreter is missing the "initializer of 'c' unknown" diagnostics.
const int c; // ref-note {{declared here}} \
             // pedantic-ref-note {{declared here}}
_Static_assert(c == 0, ""); // ref-error {{not an integral constant expression}} \
                            // ref-note {{initializer of 'c' is unknown}} \
                            // pedantic-ref-error {{not an integral constant expression}} \
                            // pedantic-ref-note {{initializer of 'c' is unknown}} \
                            // expected-error {{not an integral constant expression}} \
                            // pedantic-expected-error {{not an integral constant expression}}

_Static_assert(&c != 0, ""); // ref-warning {{always true}} \
                             // pedantic-ref-warning {{always true}} \
                             // pedantic-ref-warning {{is a GNU extension}} \
                             // expected-warning {{always true}} \
                             // pedantic-expected-warning {{always true}} \
                             // pedantic-expected-warning {{is a GNU extension}}
_Static_assert(&a != 0, ""); // ref-warning {{always true}} \
                             // pedantic-ref-warning {{always true}} \
                             // pedantic-ref-warning {{is a GNU extension}} \
                             // expected-warning {{always true}} \
                             // pedantic-expected-warning {{always true}} \
                             // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&c + 1) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                   // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&a + 100) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                     // pedantic-ref-note {{100 of non-array}} \
                                     // pedantic-expected-note {{100 of non-array}} \
                                     // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&a - 100) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                     // pedantic-expected-warning {{is a GNU extension}} \
                                     // pedantic-ref-note {{-100 of non-array}} \
                                     // pedantic-expected-note {{-100 of non-array}}
