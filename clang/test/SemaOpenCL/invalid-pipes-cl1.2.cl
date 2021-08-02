// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

void foo(read_only pipe int p);
// expected-warning@-1 {{type specifier missing, defaults to 'int'}}
// expected-error@-2 {{access qualifier can only be used for pipe and image type}}
// expected-error@-3 {{expected ')'}} expected-note@-3 {{to match this '('}}

// 'pipe' should be accepted as an identifier.
typedef int pipe;
