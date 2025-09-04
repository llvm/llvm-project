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


struct BitfieldUnion {
  union {
    struct {
      unsigned int addr : 22;
      unsigned int vf : 1;
    };
    unsigned int raw;
  };
};

struct BitfieldUnion processBitfieldUnion(struct BitfieldUnion r) {
  struct BitfieldUnion result = r;
  result.addr += 1;
  return result;
}

void testBitfieldUnionCompoundLiteral(void) {
  struct BitfieldUnion r1 = processBitfieldUnion((struct BitfieldUnion){.addr = 100, .vf = 1});
  clang_analyzer_eval(r1.addr == 101); // expected-warning{{TRUE}}
  clang_analyzer_eval(r1.vf == 1); // expected-warning{{UNKNOWN}}
  
  struct BitfieldUnion r2 = processBitfieldUnion((struct BitfieldUnion){.addr = 1});
  clang_analyzer_eval(r2.addr == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(r2.vf); // expected-warning{{UNKNOWN}}
}

struct NestedBitfields {
  struct {
    unsigned x : 16;
    unsigned y : 16;
  } inner;
};

struct NestedBitfields processNestedBitfields(struct NestedBitfields n) {
  struct NestedBitfields tmp = n;
  tmp.inner.x += 1;
  return tmp;
}

void testNestedBitfields(void) {
  struct NestedBitfields n1 = processNestedBitfields((struct NestedBitfields){.inner.x = 1});
  clang_analyzer_eval(n1.inner.x == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(n1.inner.y == 0); // expected-warning{{TRUE}}
  
  struct NestedBitfields n2 = processNestedBitfields((struct NestedBitfields){{200, 300}});
  clang_analyzer_eval(n2.inner.x == 201); // expected-warning{{TRUE}}
  clang_analyzer_eval(n2.inner.y == 300); // expected-warning{{TRUE}}
}

struct UnionContainerBitfields {
  union {
    unsigned val;
    struct {
      unsigned x : 16;
      unsigned y : 16;
    };
  } u;
};

struct UnionContainerBitfields processUnionContainer(struct UnionContainerBitfields c) {
  struct UnionContainerBitfields tmp = c;
  tmp.u.x += 1;
  return tmp;
}

void testUnionContainerBitfields(void) {
  struct UnionContainerBitfields c1 = processUnionContainer((struct UnionContainerBitfields){.u.val = 100});
  clang_analyzer_eval(c1.u.x == 101); // expected-warning{{TRUE}}
  
  struct UnionContainerBitfields c2 = processUnionContainer((struct UnionContainerBitfields){.u.x = 100});
  clang_analyzer_eval(c2.u.x == 101); // expected-warning{{TRUE}}
}

struct MixedBitfields {
  unsigned char x;
  unsigned y : 12;
  unsigned z : 20;
};

struct MixedBitfields processMixedBitfields(struct MixedBitfields m) {
  struct MixedBitfields tmp = m;
  tmp.y += 1;
  return tmp;
}

void testMixedBitfields(void) {
  struct MixedBitfields m1 = processMixedBitfields((struct MixedBitfields){.x = 100, .y = 100});
  clang_analyzer_eval(m1.x == 100); // expected-warning{{TRUE}}
  clang_analyzer_eval(m1.y == 101); // expected-warning{{TRUE}}
  
  struct MixedBitfields m2 = processMixedBitfields((struct MixedBitfields){.z = 100});
  clang_analyzer_eval(m2.y == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(m2.z == 100); // expected-warning{{TRUE}}
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
