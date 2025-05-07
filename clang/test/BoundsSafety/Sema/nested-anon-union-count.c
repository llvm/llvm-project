
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// Extracted from nested-struct-member-count.c because the error message refers to a line number, making it fragile
struct InnerAnonUnion {
    struct B {
        union {
            int len;
            float f;
        };
        int dummy;
    } hdr;
    char fam[__counted_by(hdr.len)]; // expected-error-re{{count parameter refers to union 'hdr.' of type 'union B::(anonymous at {{.*}}/BoundsSafety/Sema/nested-anon-union-count.c:10:9)'}}
};
