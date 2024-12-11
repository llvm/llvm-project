
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

void bar(unsigned x, void *__sized_by(z) y, unsigned z);

// expected-note@+1{{'filter_mode' declared here}}
void foo(unsigned filter_mode) {
	void *p = 0;
	// expected-error@+1{{use of undeclared identifier 'filterMode'}}
	bar(filterMode, p, 0);
}
