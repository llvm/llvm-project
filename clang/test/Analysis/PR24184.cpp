// UNSUPPORTED: z3
// RUN: %clang_analyze_cc1 -w -analyzer-eagerly-assume -fcxx-exceptions -analyzer-checker=core -analyzer-checker=alpha.core.PointerArithm,alpha.core.CastToStruct -analyzer-max-loop 64 -verify %s
// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-checker=cplusplus -fcxx-exceptions -analyzer-checker alpha.core.PointerArithm,alpha.core.CastToStruct -analyzer-max-loop 63 -verify %s

// These tests used to hit an assertion in the bug report. Test case from http://llvm.org/PR24184.
typedef struct {
  int cbData;
  unsigned pbData;
} CRYPT_DATA_BLOB;

typedef enum { DT_NONCE_FIXED } DATA_TYPE;
int a;
typedef int *vcreate_t(int *, DATA_TYPE, int, int);
void fn1(unsigned, unsigned) {
  char b = 0;
  for (; 1; a++, &b + a * 0)
    ;
}

vcreate_t fn2;
struct A {
  CRYPT_DATA_BLOB value;
  int m_fn1() {
    int c;
    value.pbData == 0;
    fn1(0, 0);
  }
};
struct B {
  A IkeHashAlg;
  A IkeGType;
  A NoncePhase1_r;
};
class C {
  int m_fn2(B *);
  void m_fn3(B *, int, int, int);
};
int C::m_fn2(B *p1) {
  int *d;
  int e = p1->IkeHashAlg.m_fn1();
  unsigned f = p1->IkeGType.m_fn1(), h;
  int g;
  d = fn2(0, DT_NONCE_FIXED, (char)0, p1->NoncePhase1_r.value.cbData);
  h = 0 | 0;
  m_fn3(p1, 0, 0, 0);
}

// case 2:
typedef struct {
  int cbData;
  unsigned char *pbData;
} CRYPT_DATA_BLOB_1;
typedef unsigned uint32_t;
void fn1_1(void *p1, const void *p2) { p1 != p2; }

void fn2_1(uint32_t *p1, unsigned char *p2, uint32_t p3) {
  unsigned i = 0;
  for (0; i < p3; i++)
    fn1_1(p1 + i, p2 + i * 0);
}

struct A_1 {
  CRYPT_DATA_BLOB_1 value;
  uint32_t m_fn1() {
    uint32_t a;
    if (value.pbData)
      fn2_1(&a, value.pbData, value.cbData);
    return 0;
  }
};
struct {
  A_1 HashAlgId;
} *b;
void fn3() {
  uint32_t c, d;
  d = b->HashAlgId.m_fn1();
  d << 0 | 0 | 0;
  c = 0;
  0 | 1 << 0 | 0 && b;
}

// case 3:
struct ST {
  char c;
};
char *p;
int foo1(ST);
int foo2() {
  ST *p1 = (ST *)(p);      // expected-warning{{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}
  while (p1->c & 0x0F || p1->c & 0x07)
    p1 = p1 + foo1(*p1);
}

int foo3(int *node) {
  int i = foo2();
  if (i)
    return foo2();
}
