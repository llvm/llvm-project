// RUN: %clang_cc1 -fsyntax-only -verify %s

int w = z.;  // expected-error {{use of undeclared identifier 'z'}} \
             // expected-error {{expected unqualified-id}}

int x = { y[  // expected-error {{use of undeclared identifier 'y'}} \
              // expected-note {{to match this '['}} \
              // expected-note {{to match this '{'}} \
              // expected-error {{expected ';' after top level declarator}}

// The errors below all occur on the last line of the file, so splitting them
// among multiple lines doesn't work.
// expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected '}'}}
