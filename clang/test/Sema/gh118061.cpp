// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-warning@+3 {{declaration does not declare anything}}
// expected-note@+2 {{to match this '{'}}
// expected-error@+1 {{anonymous structs and classes must be class members}}
struct {
    // expected-error@+1 {{types cannot be declared in an anonymous struct}}
    enum a1 {
        b1 = (struct c1
    // expected-error@-1 {{expected ';' after struct}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}

// expected-error@+1 {{types cannot be declared in an anonymous struct}}
struct c1 {
};

// expected-note@+1 {{to match this '{'}}
union {
    enum a2 {
        b2 = (union c2
    // expected-error@-1 {{expected ';' after union}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}

union c2 {
};

// expected-note@+2 {{to match this '{'}}
// expected-warning@+1 {{declaration does not declare anything}}
class {
    // expected-error@+1 {{types cannot be declared in an anonymous struct}}
    enum a3 {
        b3 = (class c3
    // expected-error@-1 {{expected ';' after class}}
    // expected-note@-2 {{to match this '('}}
    // expected-note@-3 {{previously declared 'public' here}}
};
// expected-error@-1 {{expected ')'}}

// expected-error@+2 {{types cannot be declared in an anonymous struct}}
// expected-error@+1 {{'c3' redeclared with 'private' access}}
class c3 {
};
// expected-error@-1 {{expected ';' after class}}
// expected-error {{expected ';' after struct}} \ expected-error {{expected ';' after union}} \ expected-error 3 {{expected '}'}}
