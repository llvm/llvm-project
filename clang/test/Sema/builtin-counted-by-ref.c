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

void g(char *);

void *test1(int size) {
  int i = 0;
  char *ref = __builtin_counted_by_ref(p->array);     // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable or used as a function argument}}

  ref = __builtin_counted_by_ref(p->array);           // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable or used as a function argument}}
  ref = (char *)(int *)(42 + &*__builtin_counted_by_ref(p->array)); // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable or used as a function argument}}
  g(__builtin_counted_by_ref(p->array));              // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable or used as a function argument}}

  *__builtin_counted_by_ref(p->array) = size;         // ok
  *__builtin_counted_by_ref(&p->array[i]) = size;     // ok

  *__builtin_counted_by_ref(q->array) = size          // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  *__builtin_counted_by_ref(&q->array[0]) = size;     // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(p->x);                     // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(&p->array[i++]);           // expected-warning {{'__builtin_counted_by_ref' argument has side-effects that will be discarded}}

  __builtin_counted_by_ref();                         // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_counted_by_ref(p->array, p->x, p->count); // expected-error {{too many arguments to function call, expected 1, have 3}}

  return __builtin_counted_by_ref(p->array);          // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable or used as a function argument}}
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
  _Static_assert(_Generic(__builtin_counted_by_ref(cp->array), char * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(sp->array), short * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(ip->array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(up->array), unsigned int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(lp->array), long * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(ulp->array), unsigned long * : 1, default : 0) == 1, "wrong return type");
}
