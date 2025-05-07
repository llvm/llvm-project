
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <stddef.h>

void foo(int * __counted_by_or_null(2) p);

void bar(int * __counted_by_or_null(1) p) {
    foo(p);
}

void assign_null_constant_count() {
    int * __counted_by_or_null(2) p = NULL;
    foo(p);
}

void assign_null_dynamic_count(int count) {
    int c = count;
    int * __counted_by_or_null(c) p = NULL;
    foo(p);
}

int * __counted_by_or_null(c) quux(int c);
void side_effect();

void reassign_coupled_decls() {
    int c = 3;
    int * __counted_by_or_null(c) p;
    p = quux(c);
    c = c;

    side_effect();

    p = quux(4);
    c = 4;

    side_effect();

    int * __counted_by_or_null(5) tmp = quux(5);
    c = 5;
    p = tmp;

    side_effect();

    p = quux(6); // expected-note{{previously assigned here}}
    // expected-error@+2{{assignment to 'int *__single __counted_by_or_null(c)' (aka 'int *__single') 'p' requires corresponding assignment to 'c'; add self assignment 'c = c' if the value has not changed}}
    // expected-error@+1{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
    p = quux(6); // expected-error{{assignments to dependent variables should not have side effects between them}}

    side_effect();

    c = 7; // expected-note{{previously assigned here}}
    // expected-error@+2{{assignment to 'c' requires corresponding assignment to 'int *__single __counted_by_or_null(c)' (aka 'int *__single') 'p'; add self assignment 'p = p' if the value has not changed}}
    // expected-error@+1{{multiple consecutive assignments to a dynamic count 'c' must be simplified; keep only one of the assignments}}
    c = 7;
}
