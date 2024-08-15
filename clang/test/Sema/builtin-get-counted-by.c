// RUN: %clang_cc1 -fsyntax-only -verify %s

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
