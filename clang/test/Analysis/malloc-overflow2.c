// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -analyzer-checker=alpha.security.MallocOverflow,unix -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -analyzer-checker=alpha.security.MallocOverflow,unix,optin.portability -DPORTABILITY -verify %s

typedef __typeof__(sizeof(int)) size_t;
extern void *malloc(size_t);
extern void free(void *ptr);

void *malloc(unsigned long s);

struct table {
  int nentry;
  unsigned *table;
  unsigned offset_max;
};

static int table_build(struct table *t) {

  t->nentry = ((t->offset_max >> 2) + 31) / 32;
  t->table = (unsigned *)malloc(sizeof(unsigned) * t->nentry); // expected-warning {{the computation of the size of the memory allocation may overflow}}

  int n;
  n = 10000;
  int *p = malloc(sizeof(int) * n); // no-warning

  free(p);
  return t->nentry;
}

static int table_build_1(struct table *t) {
  t->nentry = (sizeof(struct table) * 2 + 31) / 32;
  t->table = (unsigned *)malloc(sizeof(unsigned) * t->nentry); // no-warning
  return t->nentry;
}

void *f(int n) {
  return malloc(n * 0 * sizeof(int));
#ifdef PORTABILITY
  // expected-warning@-2{{Call to 'malloc' has an allocation size of 0 bytes}}
#endif
}
