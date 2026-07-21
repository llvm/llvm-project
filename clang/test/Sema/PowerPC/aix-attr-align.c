// off-no-diagnostics
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -verify=off -Wno-aix-compat -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -verify=off -Wno-aix-compat -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux -verify=off -fsyntax-only %s

// We do not warn on any declaration with a member aligned 16. Only when the struct is passed byval.
struct R {
  int b[8] __attribute__((aligned(16))); // no-warning
};

struct S {
  int a[8] __attribute__((aligned(8)));  // no-warning
  int b[8] __attribute__((aligned(16))); // expected-warning {{alignment of 16 bytes for a struct member is not binary compatible with IBM XL C/C++ for AIX 16.1.0 or older}}
};

struct T {
  int a[8] __attribute__((aligned(8))); // no-warning
  int b[8] __attribute__((aligned(4))); // no-warning
};

int a[8] __attribute__((aligned(8)));  // no-warning
int b[4] __attribute__((aligned(16))); // no-warning

void baz(int a, int b, int *c, int d, int *e, int f, struct S);
void jaz(int a, int b, int *c, int d, int *e, int f, struct T);
void vararg_baz(int a,...);
static void static_baz(int a, int b, int *c, int d, int *e, int f, struct S sp2) {
  a = *sp2.b + *c + *e;
}

void foo(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8,
         struct S s, struct T t) {

  baz(p1, p2, s.b, p3, b, p5, s);        // expected-note {{passing byval argument 's' with potentially incompatible alignment here}}
  jaz(p1, p2, a, p3, s.a, p5, t);        // no-note
  jaz(p1, p2, s.b, p3, b, p5, t);        // no-note
  vararg_baz(p1, p2, s.b, p3, b, p5, s); // no-note
  static_baz(p1, p2, s.b, p3, b, p5, s); // no-note
}
