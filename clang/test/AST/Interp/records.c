// UNSUPPORTED: asserts
// REQUIRES: asserts
// ^ this attempts to say "don't actually run this test", because it's broken
//
// The point of this test is to demonstrate something that ExprConstant accepts,
// but Interp rejects. I had hoped to express that as the same file with two
// sets of RUNs: one for the classic evaluator, which would be expected to
// succeed, and one for the new interpreter which would be expected to fail (so
// the overall test passes just in case the new interpreter rejects something
// that the evaluator accepts).
//
// Using `XFAIL ... *` with `not` on the expected-to-pass lines isn't appropriate,
// it seems, because that will cause the test to pass when _any_ of the RUNs
// fail.
//
// We could use a RUN that groups all four commands into a single shell
// invocation that expresses the desired logical properties, possibly negating
// and using an `XFAIL` for clarity (?), but I suspect the long-term future
// of this test file is to get out of this situation and back into the "both
// match" category anyway.
//
// RUN: %clang_cc1 -verify=ref,both -std=c99 %s
// RUN: %clang_cc1 -verify=ref,both -std=c11 %s
// RUN: %clang_cc1 -verify=ref,both -std=c2x %s
//
// RUN: %clang_cc1 -verify=expected,both -std=c99 -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=expected,both -std=c11 -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=expected,both -std=c2x -fexperimental-new-constant-interpreter %s

#pragma clang diagnostic ignored "-Wgnu-folding-constant"
#pragma clang diagnostic ignored "-Wempty-translation-unit"

#if __STDC_VERSION__ >= 201112L
#define CHECK(cond) _Static_assert((cond), "")
#else
#pragma clang diagnostic ignored "-Wextra-semi"
#define CHECK(cond)
#endif

typedef struct {
    unsigned a, b;
    char cc[2];
} s_t;

// out-of-order designated initialization
// array designated initialization
const s_t s1 = { .a = 2, .b = 4, .cc[0] = 8, .cc[1] = 16 } ;
const s_t s2 = { .b = 4, .a = 2, .cc[1] = 16, .cc[0] = 8 } ;

CHECK(s1.a == s2.a && s1.b == s2.b);
CHECK(s1.cc[0] == s2.cc[0] && s1.cc[1] == s2.cc[1]);

// nested designated initialization
const struct {
    struct { unsigned v; } inner;
} nested_designated = { .inner.v = 3 };
CHECK(nested_designated.inner.v == 3);

// mixing of designated initializers and regular initializers
// both-warning@+1 {{excess elements in array initializer}}
const s_t s3 = { {}, .b = 4, {[1]=16, 8}};
const s_t s4 = { .b = 4, {[1]=16}};

CHECK(s3.a == 0);
CHECK(s3.b == 4);
CHECK(s3.cc[0] == 0);
CHECK(s3.cc[1] == 16);

CHECK(s3.a == s4.a && s3.b == s4.b);
CHECK(s3.cc[0] == s4.cc[0] && s3.cc[1] == s4.cc[1]);

const unsigned fw = 2;
typedef struct {
    struct {
        unsigned : 4;
        unsigned ff : fw;
        unsigned : 12;
        unsigned : 12;
    } in[2];

    unsigned of : 4;
    unsigned : 0;
} bf_t;

const bf_t bf0 = { };
CHECK(bf0.in[0].ff == 0);
CHECK(bf0.in[1].ff == 0);
CHECK(bf0.of == 0);

CHECK(((bf_t){{{}, {}}, {}}).of == 0);
CHECK(((bf_t){{{}, {}}, {}}).of == 0);
CHECK(((bf_t){{{}, {1}}, {}}).in[1].ff == 1);

// out-of-order designated initialization
// array designated initialization
// nested designated initialization
// mixing of designated initializers and regular initializers
// + skipped fields (unnamed bit fields)
const bf_t bf1 = { 1, 2, 3, };
const bf_t bf2 = { { [1]=2, [0]={ 1 }}, 3, };

CHECK(bf1.in[0].ff == 1);
CHECK(bf1.in[1].ff == 2);
CHECK(bf1.of == 3);

CHECK(
    bf1.in[0].ff == bf2.in[0].ff &&
    bf1.in[1].ff == bf2.in[1].ff &&
    bf1.of == bf2.of
);

unsigned func() {
    return s1.a + s2.b + bf1.of + bf2.of;
}
