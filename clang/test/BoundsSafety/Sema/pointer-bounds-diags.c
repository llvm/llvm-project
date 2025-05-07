

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

void foo(int *unqual, int *__bidi_indexable bidi, int *__indexable fwd,
         int *__single single, int *__unsafe_indexable unsafe) {
    (void)__builtin_get_pointer_lower_bound(unqual);
    (void)__builtin_get_pointer_lower_bound(bidi);
    (void)__builtin_get_pointer_lower_bound(fwd);
    (void)__builtin_get_pointer_lower_bound(single);
    // expected-error@+1{{cannot extract the lower bound of 'int *__unsafe_indexable' because it has no bounds specification}}
    (void)__builtin_get_pointer_lower_bound(unsafe);
}
