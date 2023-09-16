// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify -std=c11 %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -pedantic -verify=pedantic-expected -std=c11 %s
// RUN: %clang_cc1 -verify=ref -std=c11 %s
// RUN: %clang_cc1 -pedantic -verify=pedantic-ref -std=c11 %s

/// expected-no-diagnostics

_Static_assert(1, "");
_Static_assert(0 != 1, "");
_Static_assert(1.0 == 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                                // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert(1 && 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                              // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert( (5 > 4) + (3 > 2) == 2, "");

/// FIXME: Should also be rejected in the new interpreter
int a = (1 == 1 ? 5 : 3);
_Static_assert(a == 5, ""); // ref-error {{not an integral constant expression}} \
                            // pedantic-ref-error {{not an integral constant expression}} \
                            // pedantic-expected-warning {{not an integer constant expression}}

const int b = 3;
_Static_assert(b == 3, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                            // pedantic-expected-warning {{not an integer constant expression}}

