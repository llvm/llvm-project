// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct A {};

struct R {
    operator const A*();
};


struct B  : R {
    operator A*();
};

struct C : B {

};


void foo(C c, int A::* pmf) {
       				// FIXME. Why so many built-in candidates?
	int i = c->*pmf; 	// expected-error {{use of overloaded operator '->*' is ambiguous}} \
				// expected-note {{built-in candidate operator->*(const struct A *, const int struct A::*)}} \
				// expected-note {{built-in candidate operator->*(const struct A *, int struct A::*)}} \
				// expected-note {{built-in candidate operator->*(struct A *, const int struct A::*)}} \
				// expected-note {{built-in candidate operator->*(struct A *, int struct A::*)}}
}

