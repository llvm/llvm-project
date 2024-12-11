

// RUN: %clang_cc1 -fsyntax-only -verify=with-checks -fbounds-safety -fbounds-safety-bringup-missing-checks=indirect_count_update %s
// RUN: %clang_cc1 -fsyntax-only -verify=without-checks -fbounds-safety -fno-bounds-safety-bringup-missing-checks=indirect_count_update %s
#include <ptrcheck.h>

// without-checks-no-diagnostics

void foo(int *__ended_by(end) start, int * end) {
    // with-checks-error@+1{{assignment to 'int *__single __ended_by(end)' (aka 'int *__single') 'start' requires corresponding assignment to 'end'; add self assignment 'end = end' if the value has not changed}}
	*start++ = 0;
}

void bar(int *__ended_by(end) start, int * end) {
    // with-checks-error@+1{{assignment to 'int *__single __ended_by(end)' (aka 'int *__single') 'start' requires corresponding assignment to 'end'; add self assignment 'end = end' if the value has not changed}}
	*(start = start+1) = 0;
}
