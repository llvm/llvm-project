

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>


struct Inner {
    int dummy;
    int len;
};
struct Outer {
    struct Inner hdr;
    char fam[__counted_by(hdr.len)]; // expected-note 2{{initialized flexible array member 'fam' is here}}
};

struct Outer a = {{0, 2}, {3,5}};
void assign_global_fam() {
    a = a; // expected-error{{-fbounds-safety forbids passing 'struct Outer' by copy because it has a flexible array member}}
    a.hdr.len = 1;
}
void assign_init_list_expr() {
    a = (struct Outer){{0,33}, {7,11}}; // expected-error{{initialization of flexible array member is not allowed}}
    a.hdr.len = 1;
}

struct Outer b = {.hdr = {.len = 22}}; // expected-error{{flexible array member is initialized with 0 elements, but count value is initialized to 22}}
struct Outer c = {.hdr = {.dummy = 44}};

struct {
    int len;
    char fam[__counted_by(len)];
} d = { .len = 2, .fam = {1,2} };

struct Outer e;
void assign_global_to_global() {
    e = a; // expected-error{{-fbounds-safety forbids passing 'struct Outer' by copy because it has a flexible array member}}
}

struct Middle {
    int dummy;
    struct Inner next;
};

struct Deep {
    struct Middle hdr;
    char fam[__counted_by(hdr.next.len)]; // expected-note 8{{initialized flexible array member 'fam' is here}}
};

// if these stop emitting errors about compile-time constants it's time to add a non-const version of each case
const struct Deep f = {{-5, {-1, 1}}, {99}};
struct Deep g = {f.hdr, {98}}; // expected-error{{initializer element is not a compile-time constant}}
struct Deep h = {{-6, a.hdr}, {97}}; // expected-error{{initializer element is not a compile-time constant}}

extern struct Inner i;
struct Deep j = {{-7, i}, {96}}; // expected-error{{initializer element is not a compile-time constant}}
struct Deep k = {{-8, {-3, f.hdr.next.len}}, f.fam[0]}; // expected-error{{initializer element is not a compile-time constant}}
struct Deep l = {{-9, {-4, (float)f.hdr.next.len}}, f.fam[0]}; // expected-error{{initializer element is not a compile-time constant}}
struct Deep m = {{-10, {-5, 1}}, f.fam}; // expected-error{{initializer element is not a compile-time constant}}
                                         // expected-error@-1{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'char const[__counted_by(hdr.next.len)]' (aka 'const char[]')}}
struct Deep n = {{-11, {-6, 2}}, .fam = f.fam}; // expected-error{{flexible array requires brace-enclosed initializer}}
struct Deep o = f; // expected-error{{initializer element is not a compile-time constant}}

struct Deep p = {{-12, {-7, 1.0}}, {95}}; // expected-error{{count 'hdr.next.len' has non-integer value '1.' of type 'double'}}

// check that we handle misformed initializer lists gracefully
struct Deep q = {{-13, {-8, i}}, {94}}; // expected-error{{initializing 'int' with an expression of incompatible type 'struct Inner'}}
struct Deep r = {i, {93}}; // expected-error{{initializing 'int' with an expression of incompatible type 'struct Inner'}}
struct Deep s = {{-14, -9, 1, 42}, {92}}; // expected-warning{{excess elements in struct initializer}}
struct Deep t = {{-15, -10, 1}, {91, 90}}; // expected-error{{flexible array member is initialized with 2 elements, but count value is initialized to 1}}
struct Deep u = {.len = 1, -16, -11, 89}; // expected-error{{field designator 'len' does not refer to any field in type 'struct Deep'}}

// check that we find the right field in more obscure initializers
struct Deep v = {-17, -12, 2, 42}; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to 2}}
struct Deep w = {{-18, .next.len = 2}, {88}}; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to 2}}

struct CastInCount {
    float len;
    char fam[__counted_by((int)len)]; // expected-note{{initialized flexible array member 'fam' is here}}
};
struct CastInCount x = {2.5, {87}}; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to 2}}

typedef union {
    int i;
    float f;
} U;
struct UnionCount {
    U len;
    char fam[__counted_by(len.i)]; // expected-error{{count parameter refers to union 'len' of type 'U'}}
                                   // expected-note@-1{{initialized flexible array member 'fam' is here}}
};
struct UnionCount y = {.len.f = 3.5, {86}}; // expected-error{{count 'len.i' has non-integer value '3.5' of type 'double'}}

struct Deep ø = {
    .hdr.next.len = 2,
    .fam = {85,84,83} // expected-error{{flexible array member is initialized with 3 elements, but count value is initialized to 2}}
};

struct Deep ã = {
    .hdr.next.len = 4,
    .fam = {82,81,80} // expected-error{{flexible array member is initialized with 3 elements, but count value is initialized to 4}}
};

struct AnonBitfield {
    int : 10;
    struct {
        int : 11;
        struct {
            int : 12;
            struct {
                int : 13;
                int asdf;
            };
            struct {
                int : 14;
                int : 15;
            };
            struct {
                int : 16;
                int nonAnon: 17;
                int len;
            };
        };
        int dummy;
    } a;
    char fam[__counted_by(a.len)]; // expected-note{{initialized flexible array member 'fam' is here}}
};

struct AnonBitfield z = { 1, {}, 2, 3, 4, {5, 6, 7}};
struct AnonBitfield å = { 1, {}, 2, 3, 4, {5, 6}}; // expected-error{{flexible array member is initialized with 2 elements, but count value is initialized to 3}}
struct AnonBitfield ä = { 1, 2, 3, 4, {5, 6}}; // expected-error{{initializer for aggregate with no elements requires explicit braces}}
                                               // expected-warning@-1{{excess elements in scalar initializer}}
struct AnonBitfield ö = { { { {1}, {}, {2, 3} }, 4}, {5, 6, 7}};
struct AnonBitfield æ = { 1, {}, 2, 3, 4, 5, 6, 7 };

struct ASDF {
    int asdf;
    struct Deep footer;
};
struct ASDF ß = {
    // This error is not a -fbounds-safety limitation, but a general clang limitation that only allows initializing top level FAMs
    .footer = { .hdr.next.len = 2, .fam = {82, 81} } // expected-error{{initialization of flexible array member is not allowed}}
};
