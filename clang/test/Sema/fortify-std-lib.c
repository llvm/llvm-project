// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -fsyntax-only %s -verify

typedef unsigned long size_t;

__attribute__((fortify_stdlib(0, 0)))
int not_anything_special(); // expected-error {{'fortify_stdlib' attribute applied to an unknown function}}

__attribute__((fortify_stdlib(4, 0))) // expected-error {{'fortify_stdlib' attribute requires integer constant between 0 and 3 inclusive}}
int sprintf(char *, const char *, ...);

__attribute__((fortify_stdlib())) // expected-error {{'fortify_stdlib' attribute requires exactly 2 arguments}}
int sprintf(char *, const char *, ...);

__attribute__((fortify_stdlib(1, 2, 3))) // expected-error {{'fortify_stdlib' attribute requires exactly 2 arguments}}
int sprintf(char *, const char *, ...);

__attribute__((fortify_stdlib(-1, 2))) // expected-error {{'fortify_stdlib' attribute requires a non-negative integral compile time constant expression}}
int sprintf(char *, const char *, ...);
