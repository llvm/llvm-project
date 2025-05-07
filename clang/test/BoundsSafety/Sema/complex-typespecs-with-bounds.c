
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// Tests the correctness of applying bounds attributes to complex type specifiers
// as well as to what extent other attributes (represented by _Nullable) are retained.

#include "complex-typespecs-with-bounds.h"
#include <ptrcheck.h>

void typeoftypes() {
    typeof((long * _Nullable) 0) __single p1;
    typeof(typeof(bar) *) __single p2;
}

struct S {
    // expected-note@+1 2{{pointer 'S::f1' declared here}}
    char * _Nullable f1;
};

void typeofexprs(struct S s) {
    // expected-error@+1{{'__single' attribute only applies to pointer arguments}}
    typeof(foo) __single p1;
    // expected-error@+1{{initializing 'typeof (foo())' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(foo()) __single p2 = foo();
    typeof(&foo) __single p3 = &foo;
    // expected-error@+1{{function pointers cannot be indexable}}
    typeof(&foo) __bidi_indexable p4;
    typeof(&foo) __unsafe_indexable p5 = &foo;

    // expected-error@+1{{initializing 'typeof (bar)' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(bar) __single p6 = bar;
    typeof(&bar) __single p7 = &bar;
    typeof(bar) * __single p8 = &bar;
    // expected-error@+1{{initializing 'typeof (bar[2]) *__single' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(bar[2]) * __single p9 = bar;
    // expected-error@+1{{initializing 'typeof (&bar[2])' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(&bar[2]) __single p10 = bar;
    // expected-error@+1{{initializing 'typeof (&*bar)' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(&*bar) __single p11 = &*bar;

    // expected-warning@+1{{initializing type 'typeof (s.f1)' (aka 'char *__bidi_indexable') with an expression of type 'char *__single _Nullable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'S::f1'}}
    typeof(s.f1) __bidi_indexable p12 = s.f1;
    // expected-warning@+1{{initializing type 'typeof (*s.f1) *__bidi_indexable' (aka 'char *__bidi_indexable') with an expression of type 'char *__single _Nullable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'S::f1'}}
    typeof(*s.f1) * __bidi_indexable p13 = s.f1;
    typeof(&*s.f1) __unsafe_indexable p14 = s.f1;
}

typedef typeof(*bar) my_t;
typedef typeof(bar) my_ptr_t;
typedef typeof(*bar) * my_manual_ptr_t;

void typedefs_of_typeof() {
    // expected-error@+1{{initializing 'my_t *__single' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    my_t * __single p1 = bar;
    // expected-error@+1{{initializing 'char *__single _Nullable' with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    my_ptr_t __single p2 = bar;
    // expected-error@+1{{initializing 'my_manual_ptr_t __single' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    my_manual_ptr_t __single p3 = bar;
    // expected-error@+1{{initializing 'my_manual_ptr_t __bidi_indexable' (aka 'char *__bidi_indexable') with an expression of incompatible type 'char *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    my_manual_ptr_t __bidi_indexable p4 = bar;
    my_manual_ptr_t __unsafe_indexable p5 = bar;
    // expected-error@+1{{assigning to 'my_manual_ptr_t __bidi_indexable' (aka 'char *__bidi_indexable') from incompatible type 'my_manual_ptr_t __unsafe_indexable' (aka 'char *__unsafe_indexable') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    p4 = p5;
}

void autotypes(void * void_param) {
    // this could probably be made to work in theory, but it can always be worked around by simply adding a '*'
    __auto_type __single p1 = bar; // expected-error{{bounds attribute '__single' cannot be applied to an undeduced type}}
    __auto_type * __single p2 = bar; // expected-error{{initializing 'char *__single' with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    __auto_type * __single p3 = &*bar; // expected-error{{initializing 'char *__single' with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}

    // check that an undeduced void pointee doesn't get around errors regarding type size
    __auto_type * __bidi_indexable p4 = void_ptr; // expected-error{{initializing 'void *__bidi_indexable' with an expression of incompatible type 'void *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    // expected-note@+1{{pointer 'p5' declared here}}
    __auto_type * __bidi_indexable p5 = void_param; // expected-error{{cannot initialize indexable pointer with type 'void *__bidi_indexable' from __single pointer to incomplete type 'void *__single'; consider declaring pointer 'p5' as '__single'}}
}

void typeofexpr_typeofexpr() {
    typeof(bar) p1;
    // expected-error@+1{{initializing 'typeof (p1)' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(p1) __single p2 = bar;
}

void typeofexpr_typeoftype_typeofexpr() {
    typeof(typeof(bar)) p1;
    // expected-error@+1{{initializing 'typeof (p1)' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(p1) __single p2 = bar;
}

void typeof_autotype1() {
    __auto_type p1 = bar;
    // expected-error@+1{{bounds attribute '__single' cannot be applied to attributed type 'char * _Nullable' in this context due to the surrounding 'typeof' specifier}}
    typeof(p1) __single p2 = bar;
    // expected-error@+1{{bounds attribute '__bidi_indexable' cannot be applied to attributed type 'char * _Nullable' in this context due to the surrounding 'typeof' specifier}}
    typeof(p1) __bidi_indexable p3 = bar;
    // expected-error@+1{{bounds attribute '__unsafe_indexable' cannot be applied to attributed type 'char * _Nullable' in this context due to the surrounding 'typeof' specifier}}
    typeof(p1) __unsafe_indexable p4 = bar;
    // expected-error@+1{{bounds attribute '__indexable' cannot be applied to attributed type 'char * _Nullable' in this context due to the surrounding 'typeof' specifier}}
    typeof(p1) __indexable p5 = bar;
}

void typeof_autotype2() {
    __auto_type * p1 = bar;
    // expected-error@+1{{initializing 'typeof (p1)' (aka 'char *__single') with an expression of incompatible type 'char * _Nullable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(p1) __single p2 = bar;
}

void typeof_autotype3() {
    __auto_type p1 = bare;
    // expected-error@+1{{initializing 'typeof (p1)' (aka 'char *__single') with an expression of incompatible type 'char *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    typeof(p1) __single p2 = bare;
}


// check that we don't emit the same error twice
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
_Atomic(int * _Nullable) __attribute__((address_space(2)))  __indexable global1;
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
int * _Atomic __attribute__((address_space(2)))  __indexable global2;
