
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

#pragma clang diagnostic ignored "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"

void Test(int sel) {
    int a;
    int *x = &a;
    int *y = &a;
    int *z = sel ? x : y;

    char c;
    char *x_char = &c;
    char *y_char = &c;
    char *z_char = sel ? x_char : y_char;

    void *__indexable x_ix_void = &a;
    void *__indexable y_ix_void = &a;
    void *__indexable z_ix_void = y ?: y;

    int *__indexable x_ix = &a;
    int *__indexable y_ix = &a;
    int *__indexable z_ix = sel ? x_ix : y_ix;

    int *__single x_sg = &a;
    int *__single y_sg = &a;
    int *__single z_sg = sel ? x_sg : y_sg;

    int *__unsafe_indexable x_uix = &a;
    int *__unsafe_indexable y_uix = &a;
    int *__unsafe_indexable z_uix = sel ? x_uix : y_uix;

    char *__unsafe_indexable x_uix_char = &c;
    char *__unsafe_indexable y_uix_char = &c;
    char *__unsafe_indexable z_uix_char = sel ? x_uix_char : y_uix_char;

    z = sel ? x : y;
    z = sel ? x : y_ix;
    z = sel ? x : y_sg;
    z = sel ? x : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x : y_ix_void;

    z = sel ? x_ix : y;
    z = sel ? x_ix : y_ix;
    z = sel ? x_ix : y_sg;
    z = sel ? x_ix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_ix : y_ix_void;

    z = sel ? x_sg : y;
    z = sel ? x_sg : y_ix;
    z = sel ? x_sg : y_sg;
    z = sel ? x_sg : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_sg : y_ix_void;

    z = sel ? x_uix : y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_uix : y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_uix : y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_uix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_uix : y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z = sel ? x_ix_void : y;
    z = sel ? x_ix_void : y_ix;
    z = sel ? x_ix_void : y_sg;
    z = sel ? x_ix_void : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = sel ? x_ix_void : y_ix_void;

    z_ix = sel ? x : y;
    z_ix = sel ? x : y_ix;
    z_ix = sel ? x : y_sg;
    z_ix = sel ? x : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x : y_ix_void;

    z_ix = sel ? x_ix : y;
    z_ix = sel ? x_ix : y_ix;
    z_ix = sel ? x_ix : y_sg;
    z_ix = sel ? x_ix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_ix : y_ix_void;

    z_ix = sel ? x_sg : y;
    z_ix = sel ? x_sg : y_ix;
    z_ix = sel ? x_sg : y_sg;
    z_ix = sel ? x_sg : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_sg : y_ix_void;

    z_ix = sel ? x_uix : y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_uix : y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_uix : y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_uix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_uix : y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_ix = sel ? x_ix_void : y;
    z_ix = sel ? x_ix_void : y_ix;
    z_ix = sel ? x_ix_void : y_sg;
    z_ix = sel ? x_ix_void : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = sel ? x_ix_void : y_ix_void;

    z_sg = sel ? x_ix : y;
    z_sg = sel ? x_ix : y_ix;
    z_sg = sel ? x_ix : y_sg;
    z_sg = sel ? x_ix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_ix : y_ix_void;

    z_sg = sel ? x_sg : y;
    z_sg = sel ? x_sg : y_ix;
    z_sg = sel ? x_sg : y_sg;
    z_sg = sel ? x_sg : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_sg : y_ix_void;

    z_sg = sel ? x_uix : y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_uix : y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_uix : y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_uix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_uix : y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_sg = sel ? x_ix_void : y;
    z_sg = sel ? x_ix_void : y_ix;
    z_sg = sel ? x_ix_void : y_sg;
    z_sg = sel ? x_ix_void : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = sel ? x_ix_void : y_ix_void;

    z_uix = sel ? x_ix : y;
    z_uix = sel ? x_ix : y_ix;
    z_uix = sel ? x_ix : y_sg;
    z_uix = sel ? x_ix : y_uix;
    z_uix = sel ? x_ix : y_ix_void;

    z_uix = sel ? x_sg : y;
    z_uix = sel ? x_sg : y_ix;
    z_uix = sel ? x_sg : y_sg;
    z_uix = sel ? x_sg : y_uix;
    z_uix = sel ? x_sg : y_ix_void;

    z_uix = sel ? x_uix : y;
    z_uix = sel ? x_uix : y_ix;
    z_uix = sel ? x_uix : y_sg;
    z_uix = sel ? x_uix : y_uix;
    z_uix = sel ? x_uix : y_ix_void;

    z_uix = sel ? x_ix_void : y;
    z_uix = sel ? x_ix_void : y_ix;
    z_uix = sel ? x_ix_void : y_sg;
    z_uix = sel ? x_ix_void : y_uix;
    z_uix = sel ? x_ix_void : y_ix_void;

    z_ix_void = sel ? x_ix : y;
    z_ix_void = sel ? x_ix : y_ix;
    z_ix_void = sel ? x_ix : y_sg;
    z_ix_void = sel ? x_ix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_ix : y_ix_void;

    z_ix_void = sel ? x_sg : y;
    z_ix_void = sel ? x_sg : y_ix;
    z_ix_void = sel ? x_sg : y_sg;
    z_ix_void = sel ? x_sg : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_sg : y_ix_void;

    z_ix_void = sel ? x_uix : y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_uix : y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_uix : y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_uix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_uix : y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_ix_void = sel ? x_ix_void : y;
    z_ix_void = sel ? x_ix_void : y_ix;
    z_ix_void = sel ? x_ix_void : y_sg;
    z_ix_void = sel ? x_ix_void : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = sel ? x_ix : y;

    z_char = sel ? x_ix : y_ix; // expected-warning{{incompatible pointer types assigning}}
    z_char = sel ? x_ix : y_sg; // expected-warning{{incompatible pointer types assigning}}
    z_char = sel ? x_ix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_ix : y_ix_void;

    z_char = sel ? x_sg : y; // expected-warning{{incompatible pointer types assigning}}
    z_char = sel ? x_sg : y_ix; // expected-warning{{incompatible pointer types assigning}}
    z_char = sel ? x_sg : y_sg; // expected-warning{{incompatible pointer types assigning}}
    z_char = sel ? x_sg : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_sg : y_ix_void;

    z_char = sel ? x_uix : y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_uix : y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_uix : y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_uix : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_uix : y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_char = sel ? x_ix_void : y;
    z_char = sel ? x_ix_void : y_ix;
    z_char = sel ? x_ix_void : y_sg;
    z_char = sel ? x_ix_void : y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = sel ? x_ix_void : y_ix_void;

    z = x ?: y;
    z = x ?: y_ix;
    z = x ?: y_sg;
    z = x ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x ?: y_ix_void;

    z = x_ix ?: y;
    z = x_ix ?: y_ix;
    z = x_ix ?: y_sg;
    z = x_ix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_ix ?: y_ix_void;

    z = x_sg ?: y;
    z = x_sg ?: y_ix;
    z = x_sg ?: y_sg;
    z = x_sg ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_sg ?: y_ix_void;

    z = x_uix ?: y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_uix ?: y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_uix ?: y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_uix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_uix ?: y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z = x_ix_void ?: y;
    z = x_ix_void ?: y_ix;
    z = x_ix_void ?: y_sg;
    z = x_ix_void ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z = x_ix_void ?: y_ix_void;

    z_ix = x ?: y;
    z_ix = x ?: y_ix;
    z_ix = x ?: y_sg;
    z_ix = x ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x ?: y_ix_void;

    z_ix = x_ix ?: y;
    z_ix = x_ix ?: y_ix;
    z_ix = x_ix ?: y_sg;
    z_ix = x_ix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_ix ?: y_ix_void;

    z_ix = x_sg ?: y;
    z_ix = x_sg ?: y_ix;
    z_ix = x_sg ?: y_sg;
    z_ix = x_sg ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_sg ?: y_ix_void;

    z_ix = x_uix ?: y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_uix ?: y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_uix ?: y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_uix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_uix ?: y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_ix = x_ix_void ?: y;
    z_ix = x_ix_void ?: y_ix;
    z_ix = x_ix_void ?: y_sg;
    z_ix = x_ix_void ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix = x_ix_void ?: y_ix_void;

    z_sg = x_ix ?: y;
    z_sg = x_ix ?: y_ix;
    z_sg = x_ix ?: y_sg;
    z_sg = x_ix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_ix ?: y_ix_void;

    z_sg = x_sg ?: y;
    z_sg = x_sg ?: y_ix;
    z_sg = x_sg ?: y_sg;
    z_sg = x_sg ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_sg ?: y_ix_void;

    z_sg = x_uix ?: y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_uix ?: y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_uix ?: y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_uix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_uix ?: y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_sg = x_ix_void ?: y;
    z_sg = x_ix_void ?: y_ix;
    z_sg = x_ix_void ?: y_sg;
    z_sg = x_ix_void ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_sg = x_ix_void ?: y_ix_void;

    z_uix = x_ix ?: y;
    z_uix = x_ix ?: y_ix;
    z_uix = x_ix ?: y_sg;
    z_uix = x_ix ?: y_uix;
    z_uix = x_ix ?: y_ix_void;

    z_uix = x_sg ?: y;
    z_uix = x_sg ?: y_ix;
    z_uix = x_sg ?: y_sg;
    z_uix = x_sg ?: y_uix;
    z_uix = x_sg ?: y_ix_void;

    z_uix = x_uix ?: y;
    z_uix = x_uix ?: y_ix;
    z_uix = x_uix ?: y_sg;
    z_uix = x_uix ?: y_uix;
    z_uix = x_uix ?: y_ix_void;

    z_uix = x_ix_void ?: y;
    z_uix = x_ix_void ?: y_ix;
    z_uix = x_ix_void ?: y_sg;
    z_uix = x_ix_void ?: y_uix;
    z_uix = x_ix_void ?: y_ix_void;

    z_ix_void = x_ix ?: y;
    z_ix_void = x_ix ?: y_ix;
    z_ix_void = x_ix ?: y_sg;
    z_ix_void = x_ix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_ix ?: y_ix_void;

    z_ix_void = x_sg ?: y;
    z_ix_void = x_sg ?: y_ix;
    z_ix_void = x_sg ?: y_sg;
    z_ix_void = x_sg ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_sg ?: y_ix_void;

    z_ix_void = x_uix ?: y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_uix ?: y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_uix ?: y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_uix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_uix ?: y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_ix_void = x_ix_void ?: y;
    z_ix_void = x_ix_void ?: y_ix;
    z_ix_void = x_ix_void ?: y_sg;
    z_ix_void = x_ix_void ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_ix_void = x_ix ?: y;

    z_char = x_ix ?: y_ix; // expected-warning{{incompatible pointer types assigning}}
    z_char = x_ix ?: y_sg; // expected-warning{{incompatible pointer types assigning}}
    z_char = x_ix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_ix ?: y_ix_void;

    z_char = x_sg ?: y; // expected-warning{{incompatible pointer types assigning}}
    z_char = x_sg ?: y_ix; // expected-warning{{incompatible pointer types assigning}}
    z_char = x_sg ?: y_sg; // expected-warning{{incompatible pointer types assigning}}
    z_char = x_sg ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_sg ?: y_ix_void;

    z_char = x_uix ?: y; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_uix ?: y_ix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_uix ?: y_sg; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_uix ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_uix ?: y_ix_void; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    z_char = x_ix_void ?: y;
    z_char = x_ix_void ?: y_ix;
    z_char = x_ix_void ?: y_sg;
    z_char = x_ix_void ?: y_uix; // expected-error-re{{assigning to '{{.+}}' from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    z_char = x_ix_void ?: y_ix_void;
}

