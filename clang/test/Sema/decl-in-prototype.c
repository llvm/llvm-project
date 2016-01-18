// RUN: %clang_cc1 -fsyntax-only -verify %s

const int AA = 5;

int f1(enum {AA,BB} E) { // expected-warning {{will not be visible outside of this function}}
    return BB;
}

int f2(enum {AA=7,BB} E) { // expected-warning {{will not be visible outside of this function}}
    return AA;
}

struct a {
};

int f3(struct a { } *); // expected-warning {{will not be visible outside of this function}}

struct A { struct b { int j; } t; }; // expected-note {{previous definition is here}}

int f4(struct A { struct b { int j; } t; } *); // expected-warning {{declaration of 'struct A' will not be visible outside of this function}} expected-warning {{redefinition of 'b' will not be visible outside of this function}}

struct aA {
    struct ab { // expected-note {{previous definition is here}} expected-note {{previous definition is here}}
        int j;
    } b;
};

int f5(struct aA { struct ab { int j; } b; struct ab { char glorx; } glorx; } *); // expected-warning {{declaration of 'struct aA' will not be visible}} expected-warning {{redefinition of 'ab' will not be visible}} expected-warning {{redefinition of 'ab' will not be visible}}

void f6(struct z {int b;} c) { // expected-warning {{declaration of 'struct z' will not be visible outside of this function}}
    struct z d;
    d.b = 4;
}

void pr19018_1 (enum e19018 { qq } x); // expected-warning{{declaration of 'enum e19018' will not be visible outside of this function}}
enum e19018 qq; //expected-error{{tentative definition has type 'enum e19018' that is never completed}} \
                //expected-note{{forward declaration of 'enum e19018'}}
