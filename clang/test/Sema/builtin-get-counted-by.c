// RUN: %clang_cc1 -fsyntax-only -verify %s

struct fam_struct {
  char x;
  short count;
  int array[] __attribute__((counted_by(count)));
} *p;

struct non_fam_struct {
  char x;
  short count;
  int array[];
} *q;

void foo(int size) {
  *__builtin_get_counted_by(p->array) = size;

  if (__builtin_get_counted_by(q->array))
    *__builtin_get_counted_by(q->array) = size;

  *__builtin_get_counted_by(p->count) = size; // expected-error{{incompatible integer to pointer conversion passing 'short' to parameter of type 'void *'}}
}
