
// RUN: cp %s %t
// RUN: not %clang_cc1 -fbounds-safety -fdiagnostics-parseable-fixits -fixit -fix-what-you-can %t > %t.cc_out 2> %t.cc_out
// RUN: grep -v FIXIT-CHECK %t | FileCheck --check-prefix=FIXIT-CHECK %s
// RUN: FileCheck --check-prefix=DPF-CHECK %s --input-file=%t.cc_out

struct foo;
#define FOOP struct foo *
typedef struct foo *foop;

void take_single(struct foo **);

void no_fixits(void) {
    // CHECK-NOT: fix-it:"{{.+}}autobound-pointers.c":{12:
    // FIXIT-CHECK: struct foo *no_fixit;
    struct foo *no_fixit; // No Fix
    take_single(&no_fixit);
}

#include <ptrcheck.h>

void take_indexable(struct foo *__indexable *);
void take_bidi_indexable(struct foo *__bidi_indexable *);
void take_unsafe_indexable(struct foo *__unsafe_indexable *);

void fixits(void) {
    // FIXIT-CHECK: struct foo *__single t_single;
    struct foo *t_single; // Fix
    // FIXIT-CHECK: struct foo *__indexable t_idx;
    struct foo *t_idx; // Fix
    // FIXIT-CHECK: struct foo *t_bidi;
    struct foo *t_bidi; // No Fix
    // FIXIT-CHECK: struct foo *__unsafe_indexable t_unsafe;
    struct foo *t_unsafe;
    take_single(&t_single);
    take_indexable(&t_idx);
    take_bidi_indexable(&t_bidi);
    take_unsafe_indexable(&t_unsafe);

    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{37:10-37:10}:"__single "
    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{37:10-37:10}:"__indexable "
    // CHECK-NOT: fix-it:"{{.+}}autobound-pointers.c":{37:10-37:10}:"__bidi_indexable "
    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{37:10-37:10}:"__unsafe_indexable "
    // FIXIT-CHECK: FOOP __single t_single_macro;
    FOOP t_single_macro; // Fix
    // FIXIT-CHECK: FOOP __indexable t_idx_macro;
    FOOP t_idx_macro; // Fix
    // FIXIT-CHECK: FOOP t_bidi_macro;
    FOOP t_bidi_macro; // No Fix
    // FIXIT-CHECK: FOOP __unsafe_indexable t_unsafe_macro;
    FOOP t_unsafe_macro; // Fix
    take_single(&t_single_macro);
    take_indexable(&t_idx_macro);
    take_bidi_indexable(&t_bidi_macro);
    take_unsafe_indexable(&t_unsafe_macro);

    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{47:10-47:10}:"__single "
    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{47:10-47:10}:"__indexable "
    // CHECK-NOT: fix-it:"{{.+}}autobound-pointers.c":{47:10-47:10}:"__bidi_indexable "
    // CHECK: fix-it:"{{.+}}autobound-pointers.c":{47:10-47:10}:"__unsafe_indexable "
    // FIXIT-CHECK: foop __single take_single_td;
    foop take_single_td; // Fix
    // FIXIT-CHECK: foop __indexable take_idx_td;
    foop take_idx_td; // Fix
    // FIXIT-CHECK: foop take_bidi_td;
    foop take_bidi_td; // No Fix
    // FIXIT-CHECK: foop __unsafe_indexable take_unsafe_td;
    foop take_unsafe_td;
    take_single(&take_single_td);
    take_indexable(&take_idx_td);
    take_bidi_indexable(&take_bidi_td);
    take_unsafe_indexable(&take_unsafe_td);

    // FIXIT-CHECK: struct foo * __single t_single2;
    struct foo * t_single2; // Fix
    // FIXIT-CHECK: struct foo * __indexable t_idx2;
    struct foo * t_idx2; // Fix
    // FIXIT-CHECK: struct foo * t_bidi2;
    struct foo * t_bidi2; // No Fix
    // FIXIT-CHECK: struct foo * __unsafe_indexable t_unsafe2;
    struct foo * t_unsafe2; // Fix
    take_single(&t_single2);
    take_indexable(&t_idx2);
    take_bidi_indexable(&t_bidi2);
    take_unsafe_indexable(&t_unsafe2);

    // FIXIT-CHECK: struct foo* __single t_single3;
    struct foo* t_single3; // Fix
    // FIXIT-CHECK:  struct foo* __indexable t_idx3
    struct foo* t_idx3; // Fix
    // FIXIT-CHECK: struct foo* t_bidi3;
    struct foo* t_bidi3; // No Fix
    // FIXIT-CHECK: struct foo* __unsafe_indexable t_unsafe3;
    struct foo* t_unsafe3; // Fix
    take_single(&t_single3);
    take_indexable(&t_idx3);
    take_bidi_indexable(&t_bidi3);
    take_unsafe_indexable(&t_unsafe3);
}

void multiple_fixes_to_decl(void) {
    // "DPF" checks (diagnostics-parseable-fixits) are used to check that the other fixits
    // that don't get applied automatically are emitted in someway.

    // FIXIT-CHECK: struct foo* __single mftd_single;
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+2]]:17-[[@LINE+2]]:17}:"__single "
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__single "
    struct foo* mftd_single; // Fix
    // FIXIT-CHECK: struct foo* __indexable mftd_idx;
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+2]]:17-[[@LINE+2]]:17}:"__indexable "
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__indexable "
    struct foo* mftd_idx; // Fix
    // FIXIT-CHECK: struct foo* mftd_bidi;
    struct foo* mftd_bidi; // No Fix
    // FIXIT-CHECK: struct foo* __unsafe_indexable mftd_unsafe;
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+2]]:17-[[@LINE+2]]:17}:"__unsafe_indexable "
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__unsafe_indexable "
    struct foo* mftd_unsafe; // Fix
    // FIXIT-CHECK: struct foo* __single mftd_mixed;
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+3]]:17-[[@LINE+3]]:17}:"__single "
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+2]]:17-[[@LINE+2]]:17}:"__indexable "
    // DPF-CHECK: fix-it:"{{.+}}autobound-pointers.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__unsafe_indexable "
    struct foo* mftd_mixed; // Fix

    take_single(&mftd_single);
    take_single(&mftd_single);

    take_indexable(&mftd_idx);
    take_indexable(&mftd_idx);

    take_bidi_indexable(&mftd_bidi);
    take_bidi_indexable(&mftd_bidi);

    take_unsafe_indexable(&mftd_unsafe);
    take_unsafe_indexable(&mftd_unsafe);

    // In this case the first call is used to generate the fix-it that
    // is automatically.
    take_single(&mftd_mixed);
    take_indexable(&mftd_mixed);
    take_bidi_indexable(&mftd_mixed);
    take_unsafe_indexable(&mftd_mixed);
}

void take_counted_by(struct foo *__sized_by(*size) *, int *size);

struct foo *glob;

void no_fixits_2(void) {
    // FIXIT-CHECK: struct foo *__bidi_indexable i;
    struct foo *__bidi_indexable i; // No Fix
    take_single(&i);

    struct foo *j;
    int size = 0;
    take_counted_by(&j, &size);

    int *k;
    take_single(&k);

    struct foo *__unsafe_indexable *l = &glob;
}
