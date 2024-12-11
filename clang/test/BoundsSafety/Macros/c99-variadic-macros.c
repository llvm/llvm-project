

// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

// expected-no-diagnostics

#include <ptrcheck.h>

void foo(void) {
    char *p;

    // This macro cannot be declared as __unsafe_null_terminated_from_indexable(P, ...),
    // because then the C standard pre-C23 doesn't allow passing only 1 argument.
    __unsafe_null_terminated_from_indexable((void)p);
    __unsafe_null_terminated_from_indexable((void)p, trailing);
    __unsafe_null_terminated_from_indexable((void)p, trailing, trailing2);

    // This macro cannot be declared as __unsafe_terminated_by_from_indexable(T, P, ...),
    // because then the C standard pre-C23 doesn't allow passing only 2 arguments.
    __unsafe_terminated_by_from_indexable(filler, (void)p);
    __unsafe_terminated_by_from_indexable(filler, (void)p, trailing);
    __unsafe_terminated_by_from_indexable(filler, (void)p, trailing, trailing2);
}
