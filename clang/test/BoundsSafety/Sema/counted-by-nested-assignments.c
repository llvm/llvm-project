

// RUN: %clang_cc1 -fsyntax-only -verify=with-checks -fbounds-safety -fbounds-safety-bringup-missing-checks=indirect_count_update %s
// RUN: %clang_cc1 -fsyntax-only -verify=without-checks -fbounds-safety -fno-bounds-safety-bringup-missing-checks=indirect_count_update %s
#include <ptrcheck.h>

// without-checks-no-diagnostics

void foo(int *__counted_by(count) x, int count) {
    // with-checks-error@+1{{assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'x' requires corresponding assignment to 'count'; add self assignment 'count = count' if the value has not changed}}
	*x++ = 0;
}

void bar(int *__counted_by(count) x, int count) {
    // with-checks-error@+1{{assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'x' requires corresponding assignment to 'count'; add self assignment 'count = count' if the value has not changed}}
	*(x = x+1) = 0;
}
