// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-warning@+2 {{declaration does not declare anything}}
// expected-error@+1 {{anonymous structs and classes must be class members}}
struct {
    // expected-error@+1 {{types cannot be declared in an anonymous struct}}
    enum a1 {
        b1 = (struct c1
    // expected-error@-1 {{expected ';' after struct}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}
// expected-error@-2 {{expected ';' after enum}}

struct c1 {
};

// expected-warning@+2 {{declaration does not declare anything}}
// expected-error@+1 {{anonymous unions at namespace or global scope must be declared 'static'}}
union {
    // expected-error@+1 {{types cannot be declared in an anonymous union}}
    enum a2 {
        b2 = (union c2
    // expected-error@-1 {{expected ';' after union}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}
// expected-error@-2 {{expected ';' after enum}}

union c2 {
};

// expected-warning@+2 {{declaration does not declare anything}}
// expected-error@+1 {{anonymous structs and classes must be class members}}
class {
    // expected-error@+1 {{types cannot be declared in an anonymous struct}}
    enum a3 {
        b3 = (class c3
    // expected-error@-1 {{expected ';' after class}}
    // expected-note@-2 {{to match this '('}}
};
// expected-error@-1 {{expected ')'}}
// expected-error@-2 {{expected ';' after enum}}

class c3 {
};
