// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef struct S1 { char c; } S1 __attribute__((aligned(8)));
static_assert(alignof(S1) == 8, "attribute ignored");
static_assert(alignof(struct S1) == 1, "attribute applied to original type");

typedef struct __attribute__((aligned(8))) S2 { char c; } AS;
static_assert(alignof(S2) == 8, "attribute not propagated");
static_assert(alignof(struct S2) == 8, "attribute ignored");

typedef struct __attribute__((aligned(4))) S3 {
  char c;
} S3 __attribute__((aligned(8)));
static_assert(alignof(S3) == 8, "attribute ignored");
static_assert(alignof(struct S3) == 4, "attribute clobbered");
