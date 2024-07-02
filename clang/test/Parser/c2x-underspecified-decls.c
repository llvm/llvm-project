// RUN: %clang_cc1 -fsyntax-only -verify=expected,c23 -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c17 -std=c17 %s

auto underspecified_struct = (struct S1 { int x, y; }){ 1, 2 };         // c23-error {{'struct S1' is defined as an underspecified object initializer}} \
                                                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                                                           c17-error {{illegal storage class on file-scoped variable}}
auto underspecified_union = (union U1 { int a; double b; }){ .a = 34 }; // c23-error {{'union U1' is defined as an underspecified object initializer}} \
                                                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                                                           c17-error {{illegal storage class on file-scoped variable}}
auto underspecified_enum = (enum E1 { FOO, BAR }){ BAR };               // c23-error {{'enum E1' is defined as an underspecified object initializer}} \
                                                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                                                           c17-error {{illegal storage class on file-scoped variable}}
