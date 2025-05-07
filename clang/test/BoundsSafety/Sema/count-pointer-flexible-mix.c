
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Mix {
    int buf_count;
    int *__counted_by(buf_count) buf;
    int flex_count;
    int fam[__counted_by(flex_count)];
};

void assign_to_buf_count1(struct Mix *p) {
    p->buf_count = 0;
    p->buf = 0;
}


void assign_to_buf_count2(struct Mix *p) {
    p->buf_count = 0; // expected-error{{assignment to 'p->buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->buf'; add self assignment 'p->buf = p->buf' if the value has not changed}}
}

void assign_to_flex_count1(struct Mix *p) {
    p->flex_count = 0; // expected-error{{assignment to 'p->flex_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
}

void assign_to_flex_count2(struct Mix *p, void *__bidi_indexable in_p, int in_count) {
    p = in_p;
    p->flex_count = in_count;
}

void assign_to_mix1(struct Mix *p, void *__bidi_indexable in_p, int in_count) {
    p = in_p;
    p->flex_count = in_count;
    p->buf_count = 0;
    p->buf = 0;
}

void assign_to_mix2(struct Mix *p, void *__bidi_indexable in_p, int in_count) {
    p = in_p;
    p->flex_count = in_count;
    p->buf_count = 0; // expected-error{{assignment to 'p->buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->buf'; add self assignment 'p->buf = p->buf' if the value has not changed}}
}

void assign_to_mix3(struct Mix *p, void *__bidi_indexable in_p, int in_count) {
    p = in_p;
    p->buf = 0; // expected-error{{assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->buf' requires corresponding assignment to 'p->buf_count'; add self assignment 'p->buf_count = p->buf_count' if the value has not changed}}
    // Below is error because the assign to p is interrupted by p->buf = 0;
    p->flex_count = in_count; // expected-error{{assignment to 'p->flex_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
}

void assign_to_mix4(struct Mix *p, void *__bidi_indexable in_p, int in_count) {
    p = in_p;
    p->buf_count = 0; // expected-error{{assignment to 'p->buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->buf'; add self assignment 'p->buf = p->buf' if the value has not changed}}
    // Below is error because the assign to p is interrupted by p->count = 0;
    p->flex_count = in_count; // expected-error{{assignment to 'p->flex_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
}


struct MixInner {
    int buf_count;
    int *__counted_by(buf_count) buf;
    int flex_count;
};

struct MixOuter {
    struct MixInner header;
    int fam[__counted_by(header.flex_count)];
};


struct MixOuterSharedCount {
    struct MixInner header;
    int fam[__counted_by(header.buf_count)];
};

void assign_to_inner_buf_count1(struct MixInner *p) {
    p->buf_count = 0; // expected-error{{assignment to 'p->buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->buf'; add self assignment 'p->buf = p->buf' if the value has not changed}}
}

void assign_to_inner_buf_count2(struct MixInner *p) {
    p->buf_count = 0;
    p->buf = 0;
}

void assign_to_header_buf_count1(struct MixOuter *p) {
    p->header.buf_count = 0; // expected-error{{assignment to 'p->header.buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->header.buf'; add self assignment 'p->header.buf = p->header.buf' if the value has not changed}}
}

void assign_to_header_buf_count2(struct MixOuter *p) {
    p->header.buf_count = 0;
    p->header.buf = 0;
}

void assign_to_header_flex_count(struct MixOuter *p) {
    p->header.flex_count = 0; // expected-error{{assignment to 'p->header.flex_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
}

void assign_to_header_shared_count1(struct MixOuterSharedCount *p) {
    // expected-error@+1{{assignment to 'p->header.buf_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
    p->header.buf_count = 0; // expected-error{{assignment to 'p->header.buf_count' requires corresponding assignment to 'int *__single __counted_by(buf_count)' (aka 'int *__single') 'p->header.buf'; add self assignment 'p->header.buf = p->header.buf' if the value has not changed}}
}

void assign_to_header_shared_count2(struct MixOuterSharedCount *p) {
    p->header.buf_count = 0; // expected-error{{assignment to 'p->header.buf_count' requires an immediately preceding assignment to 'p' with a wide pointer}}
    p->header.buf = 0;
}

struct Indirect {
    int len1;
    int len2;
    int * __counted_by(len1) p1;
    int * __counted_by(len1 + len2) p2;
    int fam[__counted_by(len2)];
};

void assign_indirect_group(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    ptr->len1 = len1;
    ptr->len2 = len2;
    ptr->p1 = p1;
    ptr->p2 = p2;
}

void assign_indirect_group_scramble(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    ptr->p1 = p1;
    ptr->p2 = p2;
    ptr->len1 = len1;
    ptr->len2 = len2;
}

void assign_indirect_group_scramble2(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    ptr->len1 = len1;
    ptr->p1 = p1;
    ptr->len2 = len2;
    ptr->p2 = p2;
}

void assign_indirect_group_len1_missing(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    ptr->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'ptr->p1' requires corresponding assignment to 'ptr->len1'; add self assignment 'ptr->len1 = ptr->len1' if the value has not changed}}
    ptr->p1 = p1;
    ptr->p2 = p2;
}

void assign_indirect_group_len1_p1_missing(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    ptr->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'ptr->p2' requires corresponding assignment to 'ptr->len1'; add self assignment 'ptr->len1 = ptr->len1' if the value has not changed}}
    ptr->p2 = p2;
}

void assign_indirect_group_p1_missing(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    struct Indirect * __single ptr = p;
    // expected-error@+1{{assignment to 'ptr->len1' requires corresponding assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'ptr->p1'; add self assignment 'ptr->p1 = ptr->p1' if the value has not changed}}
    ptr->len1 = len1;
    ptr->len2 = len2;
    ptr->p2 = p2;
}

void assign_indirect_p2_missing(struct Indirect * __bidi_indexable p, int len1, int len2, int * __counted_by(len1) p1) {
    struct Indirect * __single ptr = p;
    // expected-error@+1{{assignment to 'ptr->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'ptr->p2'; add self assignment 'ptr->p2 = ptr->p2' if the value has not changed}}
    ptr->len1 = len1;
    ptr->len2 = len2;
    ptr->p1 = p1;
}

void assign_indirect_p2_len2_missing(struct Indirect * __bidi_indexable p, int len1, int * __counted_by(len1) p1) {
    struct Indirect * __single ptr = p;
    // expected-error@+1{{assignment to 'ptr->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'ptr->p2'; add self assignment 'ptr->p2 = ptr->p2' if the value has not changed}}
    ptr->len1 = len1;
    ptr->p1 = p1;
}

void assign_indirect_p2_len2_len1_missing(struct Indirect * __bidi_indexable p, int len1, int * __counted_by(len1) p1) {
    struct Indirect * __single ptr = p;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'ptr->p1' requires corresponding assignment to 'ptr->len1'; add self assignment 'ptr->len1 = ptr->len1' if the value has not changed}}
    ptr->p1 = p1;
}

void update_indirect_group(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-note@+1{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    p->len1 = len1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p->p1 = p1;
    p->p2 = p2;
}

void update_indirect_group_scramble(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-note@+1{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    p->p1 = p1;
    p->p2 = p2;
    p->len1 = len1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
}

void update_indirect_group_scramble2(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-note@+1{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    p->len1 = len1;
    p->p1 = p1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p->p2 = p2;
}

void update_indirect_group_len1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-error@+1{{assignment to 'p->len2' requires an immediately preceding assignment to 'p' with a wide pointer}}
    p->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p1 = p1;
    p->p2 = p2;
}

void update_indirect_group_len1_p1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    // expected-error@+1{{assignment to 'p->len2' requires an immediately preceding assignment to 'p' with a wide pointer}}
    p->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p2 = p2;
}

void update_indirect_group_p1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    // expected-note@+2{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1'; add self assignment 'p->p1 = p->p1' if the value has not changed}}
    p->len1 = len1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p->p2 = p2;
}

void update_indirect_p2_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1) {
    // expected-note@+2{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2'; add self assignment 'p->p2 = p->p2' if the value has not changed}}
    p->len1 = len1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p->p1 = p1;
}

void update_indirect_p2_len2_missing(struct Indirect * __single p, int len1, int * __counted_by(len1) p1) {
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2'; add self assignment 'p->p2 = p->p2' if the value has not changed}}
    p->len1 = len1;
    p->p1 = p1;
}

void update_indirect_p2_len2_len1_missing(struct Indirect * __single p, int len1, int * __counted_by(len1) p1) {
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p1 = p1;
}


void self_assign_update_indirect_group(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    p = p;
    p->len1 = len1;
    p->len2 = len2;
    p->p1 = p1;
    p->p2 = p2;
}

void self_assign_update_indirect_group_scramble(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-note@+1{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    p->len1 = len1;
    p = p;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p->p1 = p1;
    // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
    p->p2 = p2;
}

void self_assign_update_indirect_group_scramble2(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    // expected-note@+1{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    p->p1 = p1;
    p->p2 = p2;
    p = p;
    p->len1 = len1;
    // expected-error@+2{{assignments to dependent variables should not have side effects between them}}
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
}

void self_assign_update_indirect_group_len1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1, int * __counted_by(len1 + len2) p2) {
    p = p;
    p->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p1 = p1;
    p->p2 = p2;
}

void self_assign_update_indirect_group_len1_p1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    p = p;
    p->len2 = len2;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p2 = p2;
}

void self_assign_update_indirect_group_p1_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1 + len2) p2) {
    p = p;
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1'; add self assignment 'p->p1 = p->p1' if the value has not changed}}
    p->len1 = len1;
    p->len2 = len2;
    p->p2 = p2;
}

void self_assign_update_indirect_p2_missing(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1) {
    p = p;
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2'; add self assignment 'p->p2 = p->p2' if the value has not changed}}
    p->len1 = len1;
    p->len2 = len2;
    p->p1 = p1;
}

void self_assign_update_indirect_p2_missing_scramble(struct Indirect * __single p, int len1, int len2, int * __counted_by(len1) p1) {
    // expected-note@+2{{group of dependent field assignments starts here, place assignment to 'p' before it}}
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2'; add self assignment 'p->p2 = p->p2' if the value has not changed}}
    p->len1 = len1;
    // expected-error@+1{{assignment to 'p->len2' requires an assignment to 'p' with a wide pointer immediately preceding the group of dependent field assignments}}
    p->len2 = len2;
    p = p;
    // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
    p->p1 = p1;
}

void self_assign_update_indirect_p2_len2_missing(struct Indirect * __single p, int len1, int * __counted_by(len1) p1) {
    p = p;
    // expected-error@+1{{assignment to 'p->len1' requires corresponding assignment to 'int *__single __counted_by(len1 + len2)' (aka 'int *__single') 'p->p2'; add self assignment 'p->p2 = p->p2' if the value has not changed}}
    p->len1 = len1;
    p->p1 = p1;
}

void self_assign_update_indirect_p2_len2_len1_missing(struct Indirect * __single p, int len1, int * __counted_by(len1) p1) {
    p = p;
    // expected-error@+1{{assignment to 'int *__single __counted_by(len1)' (aka 'int *__single') 'p->p1' requires corresponding assignment to 'p->len1'; add self assignment 'p->len1 = p->len1' if the value has not changed}}
    p->p1 = p1;
}
