// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

void foo(read_only pipe int p); // expected-error {{expected parameter declarator}} expected-error {{expected ')'}} expected-note {{to match this '('}}
