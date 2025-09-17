// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdint.h>

extern void f1(int *);
extern void f2(char *);

struct Ok {
  char c;
  int x;
};

struct __attribute__((packed)) Arguable {
  char c0;
  int x;
  char c1;
};

union __attribute__((packed)) UnionArguable {
  char c;
  int x;
};

typedef struct Arguable ArguableT;

struct Arguable *get_arguable(void);

void to_void(void *);
void to_intptr(intptr_t);

void g0(void) {
  {
    struct Ok ok;
    f1(&ok.x); // no-warning
    f2(&ok.c); // no-warning
  }
  {
    struct Arguable arguable;
    f2(&arguable.c0); // no-warning
    f1(&arguable.x);  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable.c1); // no-warning

    f1((int *)(void *)&arguable.x); // no-warning
    to_void(&arguable.x);           // no-warning
    void *p = &arguable.x;          // no-warning
    to_void(p);
    to_intptr((intptr_t)p);         // no-warning
  }
  {
    union UnionArguable arguable;
    f2(&arguable.c); // no-warning
    f1(&arguable.x); // expected-warning {{packed member 'x' of class or structure 'UnionArguable'}}

    f1((int *)(void *)&arguable.x);   // no-warning
    to_void(&arguable.x);             // no-warning
    to_intptr((intptr_t)&arguable.x); // no-warning
  }
  {
    ArguableT arguable;
    f2(&arguable.c0); // no-warning
    f1(&arguable.x);  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable.c1); // no-warning

    f1((int *)(void *)&arguable.x);   // no-warning
    to_void(&arguable.x);             // no-warning
    to_intptr((intptr_t)&arguable.x); // no-warning
  }
  {
    struct Arguable *arguable = get_arguable();
    f2(&arguable->c0); // no-warning
    f1(&arguable->x);  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable->c1); // no-warning

    f1((int *)(void *)&arguable->x);    // no-warning
    to_void(&arguable->c1);             // no-warning
    to_intptr((intptr_t)&arguable->c1); // no-warning
  }
  {
    ArguableT *arguable = get_arguable();
    f2(&(arguable->c0)); // no-warning
    f1(&(arguable->x));  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&(arguable->c1)); // no-warning

    f1((int *)(void *)&(arguable->x));      // no-warning
    to_void(&(arguable->c1));               // no-warning
    to_intptr((intptr_t)&(arguable->c1));   // no-warning
  }
}

struct S1 {
  char c;
  int i __attribute__((packed));
};

int *g1(struct S1 *s1) {
  return &s1->i; // expected-warning {{packed member 'i' of class or structure 'S1'}}
}

struct S2_i {
  int i;
};
struct __attribute__((packed)) S2 {
  char c;
  struct S2_i inner;
};

int *g2(struct S2 *s2) {
  return &s2->inner.i; // expected-warning {{packed member 'inner' of class or structure 'S2'}}
}

struct S2_a {
  char c;
  struct S2_i inner __attribute__((packed));
};

int *g2_a(struct S2_a *s2_a) {
  return &s2_a->inner.i; // expected-warning {{packed member 'inner' of class or structure 'S2_a'}}
}

struct __attribute__((packed)) S3 {
  char c;
  struct {
    int i;
  } inner;
};

int *g3(struct S3 *s3) {
  return &s3->inner.i; // expected-warning {{packed member 'inner' of class or structure 'S3'}}
}

struct S4 {
  char c;
  struct __attribute__((packed)) {
    int i;
  } inner;
};

int *g4(struct S4 *s4) {
  return &s4->inner.i; // expected-warning {{packed member 'i' of class or structure 'S4::struct (unnamed at}}
}

struct S5 {
  char c;
  struct {
    char c1;
    int i __attribute__((packed));
  } inner;
};

int *g5(struct S5 *s5) {
  return &s5->inner.i; // expected-warning {{packed member 'i' of class or structure 'S5::struct (unnamed at}}
}

struct __attribute__((packed, aligned(2))) AlignedTo2 {
  int x;
};

char *g6(struct AlignedTo2 *s) {
  return (char *)&s->x; // no-warning
}

struct __attribute__((packed, aligned(2))) AlignedTo2Bis {
  int x;
};

struct AlignedTo2Bis* g7(struct AlignedTo2 *s)
{
    return (struct AlignedTo2Bis*)&s->x; // no-warning
}

typedef struct {
  char c;
  int x;
} __attribute__((packed)) TypedefStructArguable;

typedef union {
  char c;
  int x;
} __attribute((packed)) TypedefUnionArguable;

typedef TypedefStructArguable TypedefStructArguableTheSecond;

int *typedef1(TypedefStructArguable *s) {
    return &s->x; // expected-warning {{packed member 'x' of class or structure 'TypedefStructArguable'}}
}

int *typedef2(TypedefStructArguableTheSecond *s) {
    return &s->x; // expected-warning {{packed member 'x' of class or structure 'TypedefStructArguable'}}
}

int *typedef3(TypedefUnionArguable *s) {
    return &s->x; // expected-warning {{packed member 'x' of class or structure 'TypedefUnionArguable'}}
}

struct S6 {
  union {
    char c;
    int x;
  } __attribute__((packed));
};

int *anonymousInnerUnion(struct S6 *s) {
  return &s->x; // expected-warning {{packed member 'x' of class or structure 'S6::union (anonymous at}}
}

struct S6a {
    int a;
    int _;
    int c;
    char __;
    int d;
} __attribute__((packed, aligned(16))) s6;

void g8(void)
{
    f1(&s6.a); // no-warning
    f1(&s6.c); // no-warning
    f1(&s6.d); // expected-warning {{packed member 'd' of class or structure 'S6a'}}
}

struct __attribute__((packed, aligned(1))) MisalignedContainee { double d; };
struct __attribute__((aligned(8))) AlignedContainer { struct MisalignedContainee b; };

struct AlignedContainer *p;
double* g9(void) {
  return &p->b.d; // no-warning
}

union OneUnion
{
    uint32_t a;
    uint32_t b:1;
};

struct __attribute__((packed)) S7 {
    uint8_t length;
    uint8_t stuff;
    uint8_t padding[2];
    union OneUnion one_union;
};

union AnotherUnion {
    long data;
    struct S7 s;
} *au;

union OneUnion* get_OneUnion(void)
{
    return &au->s.one_union; // no-warning
}

struct __attribute__((packed)) S8 {
    uint8_t data1;
    uint8_t data2;
	uint16_t wider_data;
};

#define LE_READ_2(p)					\
	((uint16_t)					\
	 ((((const uint8_t *)(p))[0]      ) |		\
	  (((const uint8_t *)(p))[1] <<  8)))

uint32_t get_wider_data(struct S8 *s)
{
    return LE_READ_2(&s->wider_data); // no-warning
}

struct S9 {
  uint32_t x;
  uint8_t y[2];
  uint16_t z;
} __attribute__((__packed__));

typedef struct S9 __attribute__((__aligned__(16))) aligned_S9;

void g10(void) {
  struct S9 x;
  struct S9 __attribute__((__aligned__(8))) y;
  aligned_S9 z;

  uint32_t *p32;
  p32 = &x.x; // expected-warning {{packed member 'x' of class or structure 'S9'}}
  p32 = &y.x; // no-warning
  p32 = &z.x; // no-warning
}

typedef struct {
  uint32_t msgh_bits;
  uint32_t msgh_size;
  int32_t msgh_voucher_port;
  int32_t msgh_id;
} S10Header;

typedef struct {
  uint32_t t;
  uint64_t m;
  uint32_t p;
  union {
    struct {
      uint32_t a;
      double z;
    } __attribute__((aligned(8), packed)) a;
    struct {
      uint32_t b;
      double z;
      uint32_t a;
    } __attribute__((aligned(8), packed)) b;
  };
} __attribute__((aligned(8), packed)) S10Data;

typedef struct {
  S10Header hdr;
  uint32_t size;
  uint8_t count;
  S10Data data[] __attribute__((aligned(8)));
} __attribute__((aligned(8), packed)) S10;

void g11(S10Header *hdr);
void g12(S10 *s) {
  g11(&s->hdr); // no-warning
}

struct S11 {
  uint32_t x;
} __attribute__((__packed__));

void g13(void) {
  struct S11 __attribute__((__aligned__(4))) a[4];
  uint32_t *p32;
  p32 = &a[0].x; // no-warning
}

struct Invalid0 {
  void *x;
  struct fwd f; // expected-error {{incomplete type}} expected-note {{forward declaration}}
} __attribute__((packed));

void *g14(struct Invalid0 *ivl) {
  return &(ivl->x);
}

void to_void_with_expr(void *ptr, int expr);

void g15(void) {
  struct Arguable arguable;
  to_void_with_expr(&arguable.x, 3); // no-warning
  to_void_with_expr(&arguable.x, ({3;})); // no-warning
}
