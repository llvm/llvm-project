// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

struct A {};

struct B {
	operator A*();
};

struct C : B {

};


void foo(C c, B b, int A::* pmf) {
	int j = c->*pmf; 
	int i = b->*pmf;
}

struct D {
 operator const D *();
};

struct DPtr {
 operator volatile int D::*();
};

int test(D d, DPtr dptr) {
 return d->*dptr;
}

