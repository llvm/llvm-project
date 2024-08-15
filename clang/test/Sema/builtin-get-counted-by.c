// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s

struct fam_struct {
  int x;
  char count;
  int array[] __attribute__((counted_by(count)));
} *p;

struct non_fam_struct {
  char x;
  int array[42];
  short count;
} *q;

void test1(int size) {
  int i = 0;

  *__builtin_get_counted_by(p->array) = size;         // ok
  *__builtin_get_counted_by(&p->array[i]) = size;     // ok

  *__builtin_get_counted_by(q->array) = size          // expected-error {{'__builtin_get_counted_by' argument must reference a flexible array member}}
  *__builtin_get_counted_by(&q->array[0]) = size;     // expected-error {{'__builtin_get_counted_by' argument must reference a flexible array member}}
  __builtin_get_counted_by(p->x);                     // expected-error {{'__builtin_get_counted_by' argument must reference a flexible array member}}
  __builtin_get_counted_by(&p->array[i++]);           // expected-warning {{'__builtin_get_counted_by' argument has side-effects that will be discarded}}

  __builtin_get_counted_by();                         // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_get_counted_by(p->array, p->x, p->count); // expected-error {{too many arguments to function call, expected 1, have 3}}
}

struct char_count {
  char count;
  int array[] __attribute__((counted_by(count)));
} *cp;

struct short_count {
  short count;
  int array[] __attribute__((counted_by(count)));
} *sp;

struct int_count {
  int count;
  int array[] __attribute__((counted_by(count)));
} *ip;

struct unsigned_count {
  unsigned count;
  int array[] __attribute__((counted_by(count)));
} *up;

struct long_count {
  long count;
  int array[] __attribute__((counted_by(count)));
} *lp;

struct unsigned_long_count {
  unsigned long count;
  int array[] __attribute__((counted_by(count)));
} *ulp;

void test2(void) {
  _Static_assert(_Generic(__builtin_get_counted_by(cp->array), char * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_get_counted_by(sp->array), short * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_get_counted_by(ip->array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_get_counted_by(up->array), unsigned int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_get_counted_by(lp->array), long * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_get_counted_by(ulp->array), unsigned long * : 1, default : 0) == 1, "wrong return type");
}
