
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void side_effect();

void ext(int *p);

// this pointer can never be created in safe -fbounds-safety code since it requires taking the pointer of an array with __counted_by
// expected-error@+1{{pointer to incomplete __counted_by array type 'int[]' not allowed; did you mean to use a nested pointer type?}}
void ext2(int  (*ptr)[__counted_by(*len)], int *len) {
    *len = 2; // expected-error{{assignment to '*len' requires corresponding assignment to 'int[__counted_by(*len)]' (aka 'int[]') '*ptr'; add self assignment '*ptr = *ptr' if the value has not changed}}
}

void ext3(char *__counted_by(*len) ptr, int *len);
void ext4(void *p);

struct SimpleInner {
    int dummy;
    int len;
};

void ext_set_len(struct SimpleInner * p) {
    p->len = 2; // len is not referred to by any counted_by attributes yet
}

struct SimpleOuter {
    struct SimpleInner hdr;
    char fam[__counted_by(hdr.len)]; // expected-note 17{{referred to by count parameter here}}
};

void set_len(struct SimpleInner * p) {
    p->len = 2; // struct SimpleInner doesn't contain any FAMs, so not checked
                // we cannot take the address of SimpleOuter::hdr though, so this is never referred to by a FAM

    side_effect();

    p->len++;
}

void set_len2(struct SimpleOuter * s) {
    s->hdr.len = 2; // expected-error{{assignment to 's->hdr.len' requires an immediately preceding assignment to 's' with a wide pointer}}

    side_effect();

    s->hdr.len++; // expected-error{{assignment to 's->hdr.len' requires an immediately preceding assignment to 's' with a wide pointer}}
}

struct SimpleInner2 { // avoid sharing with previous test cases
    int dummy;
    int len;
};
struct SimpleOuter2 {
    struct SimpleInner2 hdr;
    char fam[__counted_by(hdr.len)];
};
struct UnrelatedFAMOuter {
    struct SimpleInner hdr;
    int dummy[];
};
struct UnrelatedCountedByFAMOuter {
    struct SimpleInner hdr;
    int outer_len;
    char fam[__counted_by(outer_len)];
};

// regression tests for assignment to count in struct with FAM that doesn't refer to the count (but some other struct does)
void set_len3(struct UnrelatedFAMOuter * s) {
    s->hdr.len = 1;
}
void set_len4() {
    struct UnrelatedFAMOuter s;
    s.hdr.len = 1;
}
void set_len5(struct UnrelatedCountedByFAMOuter * s) {
    s->hdr.len = 1;
}
void set_len6() {
    struct UnrelatedCountedByFAMOuter s;
    s.hdr.len = 1;
}

void address_of_nested_len(struct SimpleInner * p) {
    (void)&p->len;
    int * p2 = &p->len; // ok, base struct contains no FAM
}

void address_of_len(struct SimpleOuter * s) {
    int * p = &s->hdr.len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    int * p2 = &(s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    int * p3 = &(s->hdr).len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext(&s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext2(&s->fam, &s->hdr.len); // expected-error{{cannot take address of incomplete __counted_by array}}
                                // expected-error@-1{{cannot take address of field referred to by __counted_by on a flexible array member}}
                                // expected-note@-2{{remove '&' to get address as 'char *' instead of 'char (*)[__counted_by(hdr.len)]'}}

    ext2((int (*__single)[__counted_by(*len)]) &s->fam, &s->hdr.len); // expected-error{{cannot take address of incomplete __counted_by array}}
                                                                      // expected-note@-1{{remove '&' to get address as 'char *' instead of 'char (*)[__counted_by(hdr.len)]'}}
                                                                      // expected-error@-2{{cannot take address of field referred to by __counted_by on a flexible array member}}

    // type confusion
    ext2(s->fam, &s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext3(s->fam, &s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext4(&s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
                     
    (void) &s->hdr.len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    (void) &(s->hdr.len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
}

void address_of_inner_struct(struct SimpleOuter * s) {
    (void) &s->hdr; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    (void) &(s->hdr); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext4((void*)&s->hdr); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    // these are valid casts in plain C, but result in unsafe assignment to s->hdr.len due to type confusion in the -fbounds-safety type system
    set_len((struct SimpleInner *)s);
    set_len((struct SimpleInner *)&s->hdr.dummy);
}

void assign_inner_struct(struct SimpleOuter * s, struct SimpleInner * i) {
    struct SimpleInner * p = &s->hdr; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    s->hdr = (struct SimpleInner) {0, 10}; // expected-error{{cannot assign 's->hdr' because it contains field 'len' referred to by flexible array member 'fam'}}

    *&s->hdr = *i; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
}

struct InnerFAM {
    int len;
    int inner_fam[__counted_by(len)]; // expected-note{{previous use is here}}
};

void ext_set_inner_len(struct InnerFAM * p) {
    p = p;
    p->len = 2;
}

struct MultipleFAMs {
    // rdar://132712477 hdr.len is not bounds checked with respect to hdr.inner_fam, this should be an error
    struct InnerFAM hdr; // expected-warning{{field 'hdr' with variable sized type 'struct InnerFAM' not at the end of a struct or class is a GNU extension}}
    int offset;
    char fam[__counted_by(hdr.len - offset)]; // expected-error{{field 'len' referred to by flexible array member cannot be used in other dynamic bounds attributes}}
};

void set_inner_len(struct InnerFAM * p) {
    p = p;
    p->len = 2;
}

void set_len_via_fam(struct MultipleFAMs * s) {
    s->hdr.inner_fam[0] = 2; // unsafe write to s->offset
    s->hdr.inner_fam[1] = 2;
}

struct InnerInner {
    int dummy;
    int innermost_len;
};
struct MediumInner {
    struct InnerInner next;
    int dummy;
};
struct DeepNestingOuter {
    struct MediumInner hdr;
    int outer_len;
    char fam[__counted_by(outer_len + hdr.next.innermost_len)]; // expected-note 10{{referred to by count parameter here}}
};

void addresses_with_deep_nesting(struct DeepNestingOuter * s) {
    int * p = &s->outer_len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    struct MediumInner * p2 = &s->hdr; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    struct InnerInner * p3 = &s->hdr.next; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    int * p4 = &s->hdr.next.innermost_len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

    ext4(&s->outer_len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    ext4(&s->hdr); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    ext4(&s->hdr.next); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    ext4(&s->hdr.next.innermost_len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
}

void assign_with_deep_nesting(struct DeepNestingOuter * s, int a, struct MediumInner b, struct InnerInner c) {
    s->outer_len = a; // expected-error{{assignment to 's->outer_len' requires an immediately preceding assignment to 's' with a wide pointer}}
    s->hdr = b; // expected-error{{cannot assign 's->hdr' because it contains field 'next' referred to by flexible array member 'fam'}}
    s->hdr.next = c; // expected-error{{cannot assign 's->hdr.next' because it contains field 'innermost_len' referred to by flexible array member 'fam'}}
    s->hdr.next.innermost_len = a;

    side_effect();

    s = s;
    s->outer_len = a; // expected-error{{assignment to 'int' 's->outer_len' requires corresponding assignment to 's->hdr.next.innermost_len'; add self assignment 's->hdr.next.innermost_len = s->hdr.next.innermost_len' if the value has not changed}}

    side_effect();

    s = s;
    s->outer_len = a;
    s->hdr.next.innermost_len = a;
}

struct InnerDup {
    int dummy;
    int len;
};
struct OuterDup {
    struct InnerDup a;
    struct InnerDup b;
    char fam[__counted_by(a.len)]; // expected-error{{cannot depend on nested field 'len' because it exists in multiple instances in struct 'OuterDup'}}
};

void non_aliasing_assignment_to_same_decl(struct OuterDup * s) {
    s->b.len = 2; // should not affect fam

    side_effect();

    s = s;
    s->a.len = 2; // should not require assignment to s->b.len
}

struct OuterDup2 {
    struct InnerDup a;
    struct InnerDup b;
    char fam[__counted_by(a.len + b.len)]; // expected-error{{cannot depend on nested field 'len' because it exists in multiple instances in struct 'OuterDup2'}}
};

void non_aliasing_ref_to_same_decl(struct OuterDup2 * s) {
    s = s;
    s->a.len = 2; // should require assignment to s->b.len if OuterDup2::fam is error free

    side_effect();

    s = s;
    s->b.len = 2; // should require assignment to s->a.len if OuterDup2::fam is error free

    side_effect();

    s = s;
    s->a.len = 2;
    s->b.len = 2;
}

struct CommonHeader {
    int dummy;
    int len;
};
struct SubType1 {
    struct CommonHeader hdr;
    char fam1[__counted_by(hdr.len)]; // expected-note{{referred to by count parameter here}}
};
struct SubType2 {
    struct CommonHeader hdr;
    int offset;
    char fam2[__counted_by(hdr.len - offset)]; // expected-note{{referred to by count parameter here}}
};

void shared_header_type(struct SubType1 * s1, struct SubType2 * s2) {
    s1 = s1;
    s1->hdr.len = 2;

    side_effect();

    s1->hdr.len = 2; // expected-error{{assignment to 's1->hdr.len' requires an immediately preceding assignment to 's1' with a wide pointer}}

    side_effect();

    s2 = s2;
    s2->hdr.len = 2;
    s2->offset = 1;

    side_effect();

    s2 = s2;
    s2->hdr.len = 2; // expected-error{{assignment to 'int' 's2->hdr.len' requires corresponding assignment to 's2->offset'; add self assignment 's2->offset = s2->offset' if the value has not changed}}

    side_effect();

    s2 = s2;
    s2->offset = 1; // expected-error{{assignment to 'int' 's2->offset' requires corresponding assignment to 's2->hdr.len'; add self assignment 's2->hdr.len = s2->hdr.len' if the value has not changed}}

    side_effect();

    s1->hdr = s2->hdr; // expected-error{{cannot assign 's1->hdr' because it contains field 'len' referred to by flexible array member 'fam1'}}

    side_effect();

    s2->hdr = s1->hdr; // expected-error{{cannot assign 's2->hdr' because it contains field 'len' referred to by flexible array member 'fam2'}}
}

struct CommonOuterHeader {
    struct CommonHeader next;
    int dummy;
};
struct DeepSubType1 {
    struct CommonOuterHeader hdr;
    char fam1[__counted_by(hdr.next.len)]; // expected-note 2{{referred to by count parameter here}}
};
struct DeepSubType2 {
    struct CommonOuterHeader hdr;
    int offset;
    char fam2[__counted_by(hdr.next.len - offset)]; // expected-note 2{{referred to by count parameter here}}
};

void shared_nested_header_type(struct DeepSubType1 * s1, struct DeepSubType2 * s2) {
    s1 = s1;
    s1->hdr.next.len = 2;

    side_effect();

    s1->hdr.next.len = 2; // expected-error{{assignment to 's1->hdr.next.len' requires an immediately preceding assignment to 's1' with a wide pointer}}

    side_effect();

    s2 = s2;
    s2->hdr.next.len = 2;
    s2->offset = 1;

    side_effect();

    s2 = s2;
    s2->hdr.next.len = 2; // expected-error{{assignment to 'int' 's2->hdr.next.len' requires corresponding assignment to 's2->offset'; add self assignment 's2->offset = s2->offset' if the value has not changed}}

    side_effect();

    s2 = s2;
    s2->offset = 1; // expected-error{{assignment to 'int' 's2->offset' requires corresponding assignment to 's2->hdr.next.len'; add self assignment 's2->hdr.next.len = s2->hdr.next.len' if the value has not changed}}

    side_effect();

    s1->hdr = s2->hdr; // expected-error{{cannot assign 's1->hdr' because it contains field 'next' referred to by flexible array member 'fam1'}}

    side_effect();

    s2->hdr = s1->hdr; // expected-error{{cannot assign 's2->hdr' because it contains field 'next' referred to by flexible array member 'fam2'}}

    side_effect();

    s1->hdr.next = s2->hdr.next; // expected-error{{cannot assign 's1->hdr.next' because it contains field 'len' referred to by flexible array member 'fam1'}}

    side_effect();

    s2->hdr.next = s1->hdr.next; // expected-error{{cannot assign 's2->hdr.next' because it contains field 'len' referred to by flexible array member 'fam2'}}
};

struct PointerHeader {
    int dummy;
    int len;
};
struct CountPointer {
    struct PointerHeader hdr;
    char * __counted_by(hdr.len) p; // expected-error{{invalid argument expression to bounds attribute}}
                                    // expected-note@-1{{nested struct member in count parameter only supported for flexible array members}}
};
struct CountPointer2 {
    struct PointerHeader hdr;
    char * __counted_by(hdr.len) p2; // expected-error{{invalid argument expression to bounds attribute}}
                                     // expected-note@-1{{nested struct member in count parameter only supported for flexible array members}}
};

void nested_count_pointer_count(struct CountPointer * s) {
    s->hdr.len = 2;
}

struct InnerPointer {
    int len;
    char * __sized_by(len) p;
};
struct OuterWithInnerPointer {
    struct InnerPointer hdr;
    char fam[__counted_by(hdr.len)];
};

void fam_with_inner_pointer(struct OuterWithInnerPointer * s) {
    s = s;
    s->hdr.len = 2; // expected-error{{assignment to 's->hdr.len' requires corresponding assignment to 'char *__single __sized_by(len)' (aka 'char *__single') 's->hdr.p'; add self assignment 's->hdr.p = s->hdr.p' if the value has not changed}}
}

struct PointeeHeader {
    int dummy;
    int len;
};
struct FAMArrow {
    struct PointeeHeader *hdr;
    char fam[__counted_by(hdr->len)]; // expected-error{{arrow notation not allowed for struct member in count parameter}}
};
struct FAMDerefStruct {
    struct PointeeHeader *hdr;
    char fam[__counted_by((*hdr).len)]; // expected-error{{dereference operator in '__counted_by' is only allowed for function parameters}}
};

typedef union {
    int i;
    float f;
} U;
struct UnionCount {
    U len;
    char fam[__counted_by(len.i)]; // expected-error{{count parameter refers to union 'len' of type 'U'}}
                                   // expected-note@-1 2{{referred to by count parameter here}}
};
void union_assign(struct UnionCount * s) {
    s->len.i = 1;
}
void type_pun_assign(struct UnionCount * s) {
    s->len.f = 1.0;
}
void ext_u(U * u) {
    u->i = 1;
}
void union_address_param(struct UnionCount * s) {
    ext_u(&s->len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
}
void union_address_assign(struct UnionCount * s) {
    U * u  = &s->len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
    u->i = 1;
}
void ext_f(float * f) {
    *f = 1.0;
}
void union_member_address_param(struct UnionCount * s) {
    ext_f(&s->len.f);
}

struct AnonStruct {
    struct {
        int len;
        int dummy;
    };
    char fam[__counted_by(len)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}} rdar://125394428
};
struct AnonUnion {
    union {
        int len;
        float f;
    };
    char fam[__counted_by(len)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
};
struct InnerAnonStruct {
    struct A {
        struct {
            int len;
            int dummy2;
        };
        int dummy;
    } hdr;
    char fam[__counted_by(hdr.len)];
};
