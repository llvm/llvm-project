
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

typedef int * unspec_ptr_t;
typedef int *__single single_ptr_t;
typedef int *__bidi_indexable bidi_ptr_t;

unspec_ptr_t __bidi_indexable g_bidi;

void foo(unspec_ptr_t __bidi_indexable a_bidi_ok) {
    unspec_ptr_t __single l_single_ok;
    single_ptr_t __single l_single_single_ok;
    // expected-error@+1{{pointer cannot have more than one bound attribute}}
    single_ptr_t __bidi_indexable l_single_bidi_fail;
}
