// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s -Wpre-c2x-compat
// RUN: %clang_cc1 -fsyntax-only -verify=c17 -std=c17 %s

constexpr int a = 0; // c17-error {{unknown type name 'constexpr'}} \
                        c23-warning {{'constexpr' is incompatible with C standards before C23}}

void func(int array[constexpr]); // c23-error {{expected expression}} \
                                 // c17-error {{use of undeclared}}

_Atomic constexpr int b = 0; // c23-error {{constexpr variable cannot have type 'const _Atomic(int)'}} \
                             // c23-warning {{'constexpr' is incompatible with C standards before C23}} \
                             // c17-error {{unknown type name 'constexpr'}}

int static constexpr c = 1; // c17-error {{expected ';' after top level declarator}} \
                            // c23-warning {{'constexpr' is incompatible with C standards before C23}}
