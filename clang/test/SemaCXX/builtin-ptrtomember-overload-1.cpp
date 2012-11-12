// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct A {};
struct E {};

struct R {
    operator A*();
    operator E*();	// expected-note{{candidate function}}
};


struct S {
    operator A*();
    operator E*();	// expected-note{{candidate function}}
};

struct B  : R {
    operator A*();
};

struct C : B {

};

void foo(C c, int A::* pmf) {
	int i = c->*pmf; 
}

struct B1  : R, S {
    operator A*();
};

struct C1 : B1 {

};

void foo1(C1 c1, int A::* pmf) {
        int i = c1->*pmf;
        c1->*pmf = 10;
}

void foo1(C1 c1, int E::* pmf) {
        int i = c1->*pmf;	// expected-error {{use of overloaded operator '->*' is ambiguous}} \
                                // expected-note {{because of ambiguity in conversion of 'C1' to 'E *'}} \
                                // expected-note 4 {{built-in candidate operator}}
}
