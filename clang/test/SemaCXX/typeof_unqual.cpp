// RUN: %clang_cc1 -fsyntax-only -verify %s

typeof_unqual(int) u = 12; // expected-error {{expected function body after function declarator}}
__typeof_unqual(int) _u = 12;
__typeof_unqual__(int) __u = 12;
