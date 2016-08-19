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

struct Arguable *get_arguable();

void to_void(void *);

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
    void *p = &arguable.x;          // no-warning;
    to_void(p);
  }
  {
    union UnionArguable arguable;
    f2(&arguable.c); // no-warning
    f1(&arguable.x); // expected-warning {{packed member 'x' of class or structure 'UnionArguable'}}

    f1((int *)(void *)&arguable.x); // no-warning
    to_void(&arguable.x);           // no-warning
  }
  {
    ArguableT arguable;
    f2(&arguable.c0); // no-warning
    f1(&arguable.x);  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable.c1); // no-warning

    f1((int *)(void *)&arguable.x); // no-warning
    to_void(&arguable.x);           // no-warning
  }
  {
    struct Arguable *arguable = get_arguable();
    f2(&arguable->c0); // no-warning
    f1(&arguable->x);  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable->c1); // no-warning

    f1((int *)(void *)&arguable->x); // no-warning
    to_void(&arguable->c1);          // no-warning
  }
  {
    ArguableT *arguable = get_arguable();
    f2(&(arguable->c0)); // no-warning
    f1(&(arguable->x));  // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&(arguable->c1)); // no-warning

    f1((int *)(void *)&(arguable->x)); // no-warning
    to_void(&(arguable->c1));          // no-warning
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
  return &s4->inner.i; // expected-warning {{packed member 'i' of class or structure 'S4::(anonymous)'}}
}

struct S5 {
  char c;
  struct {
    char c1;
    int i __attribute__((packed));
  } inner;
};

int *g5(struct S5 *s5) {
  return &s5->inner.i; // expected-warning {{packed member 'i' of class or structure 'S5::(anonymous)'}}
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
