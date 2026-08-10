// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s
// PR23086

__builtin_isinf(...); // expected-error {{type specifier missing, defaults to 'int'}} expected-error {{ISO C requires a named parameter before '...'}} // expected-error {{cannot redeclare builtin function '__builtin_isinf'}} // expected-note {{'__builtin_isinf' is a builtin with type 'int ()'}}
