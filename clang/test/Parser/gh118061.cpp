// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-warning@+2 {{declaration does not declare anything}}
// expected-error@+1 {{anonymous structs and classes must be class members}}
struct {
    // expected-error@+1 {{types cannot be declared in an anonymous struct}}
    enum b {
        c = (struct d
    // expected-error@-1 {{expected ';' after struct}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}
// expected-error@-2 {{expected ';' after enum}}

struct d {
};
