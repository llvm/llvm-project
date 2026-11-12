// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection %s -verify

void clang_analyzer_eval(int);

unsigned foo(void);
typedef struct bf { unsigned x:2; } bf;
void bar(void) {
  bf y;
  *(unsigned*)&y = foo();
  y.x = 1;
}

struct s {
  int n;
};

void f(void) {
  struct s a;
  int *p = &(a.n) + 1; // expected-warning{{Pointer arithmetic on}}
}

typedef struct {
  int x,y;
} Point;

Point getit(void);
void test(void) {
  Point p;
  (void)(p = getit()).x;
}

#define true ((bool)1)
#define false ((bool)0)
typedef _Bool bool;


void testLazyCompoundVal(void) {
  Point p = {42, 0};
  Point q;
  clang_analyzer_eval((q = p).x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(q.x == 42); // expected-warning{{TRUE}}
}


struct Bits {
  unsigned a : 1;
  unsigned b : 2;
  unsigned c : 1;

  bool x;

  struct InnerBits {
    bool y;

    unsigned d : 16;
    unsigned e : 6;
    unsigned f : 2;
  } inner;
};

void testBitfields(void) {
  struct Bits bits;

  if (foo() && bits.b) // expected-warning {{garbage}}
    return;
  if (foo() && bits.inner.e) // expected-warning {{garbage}}
    return;

  bits.c = 1;
  clang_analyzer_eval(bits.c == 1); // expected-warning {{TRUE}}

  if (foo() && bits.b) // expected-warning {{garbage}}
    return;
  if (foo() && bits.x) // expected-warning {{garbage}}
    return;

  bits.x = true;
  clang_analyzer_eval(bits.x == true); // expected-warning{{TRUE}}
  bits.b = 2;
  clang_analyzer_eval(bits.x == true); // expected-warning{{TRUE}}
  if (foo() && bits.c) // no-warning
    return;

  bits.inner.e = 50;
  if (foo() && bits.inner.e) // no-warning
    return;
  if (foo() && bits.inner.y) // expected-warning {{garbage}}
    return;
  if (foo() && bits.inner.f) // expected-warning {{garbage}}
    return;

  extern struct InnerBits getInner(void);
  bits.inner = getInner();
  
  if (foo() && bits.inner.e) // no-warning
    return;
  if (foo() && bits.inner.y) // no-warning
    return;
  if (foo() && bits.inner.f) // no-warning
    return;

  bits.inner.f = 1;
  
  if (foo() && bits.inner.e) // no-warning
    return;
  if (foo() && bits.inner.y) // no-warning
    return;
  if (foo() && bits.inner.f) // no-warning
    return;

  if (foo() && bits.a) // expected-warning {{garbage}}
    return;
}


//-----------------------------------------------------------------------------
// Incorrect behavior
//-----------------------------------------------------------------------------

void testTruncation(void) {
  struct Bits bits;
  bits.c = 0x11; // expected-warning{{implicit truncation}}
  // FIXME: We don't model truncation of bitfields.
  clang_analyzer_eval(bits.c == 1); // expected-warning {{FALSE}}
}
