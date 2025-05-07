
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  int *__counted_by(len) buf;
  int len;
};

struct T {
  int *__counted_by(len + 1) buf;
  int len;
};

struct U {
  int *__counted_by(len) buf;
  int *__counted_by(len) buf2;
  int len;
};

struct V {
  int len;
  int buf[__counted_by(len)]; // expected-note 8{{referred to by count parameter here}}
};

int arr[10];

// expected-note@+1{{passing argument to parameter 'out_buf' here}}
void foo(int *out_len, int *__counted_by(*out_len) * out_buf) {
  *out_len = 9;
  *out_buf = arr;
  return;
}

void bar(int *fake_out_len, int **fake_out_buf) {
  *fake_out_buf = arr;
  *fake_out_len = 12;
  return;
}

void * baz(struct V *v) {
  int *__single ptr_to_len = &v->len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
  int **ptr_to_buf = &v->buf;         // expected-warning{{incompatible pointer types initializing 'int *__single*__bidi_indexable' with an expression of type 'int (*__bidi_indexable)[__counted_by(len)]' (aka 'int (*__bidi_indexable)[]')}}
                                      // expected-error@-1{{cannot take address of incomplete __counted_by array}}
                                      // expected-note@-2{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  int **ptr_ptr_to_len = &ptr_to_len;
  ptr_to_buf = &v->buf;  // expected-warning{{incompatible pointer types assigning to 'int *__single*__bidi_indexable' from 'int (*__bidi_indexable)[__counted_by(len)]' (aka 'int (*__bidi_indexable)[]')}}
                         // expected-error@-1{{cannot take address of incomplete __counted_by array}}
                         // expected-note@-2{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  *ptr_to_len = &v->len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
  *ptr_to_len = &(*v).len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}

  *ptr_to_len = 100;

  foo(&v->len, &v->buf); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
                         // expected-error@-1{{cannot take address of incomplete __counted_by array}}
                         // expected-note@-2{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  int local_len = 10;
  foo(&local_len, &v->buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
                            // expected-error@-1{{cannot take address of incomplete __counted_by array}}
                            // expected-warning@-2{{incompatible pointer types passing 'int (*__bidi_indexable)[__counted_by(len)]' (aka 'int (*__bidi_indexable)[]') to parameter of type 'int *__single __counted_by(*out_len)*__single' (aka 'int *__single*__single')}}
                            // expected-note@-3{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  bar(&v->len, &v->buf); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
                         // expected-error@-1{{cannot take address of incomplete __counted_by array}}
                         // expected-note@-2{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  (void) &v->len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
  (void) &v->buf; // expected-error{{cannot take address of incomplete __counted_by array}}
                  // expected-note@-1{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  (void) &(v->len); // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
  (void) &(v->buf); // expected-error{{cannot take address of incomplete __counted_by array}}
                  // expected-note@-1{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  return &v->len; // expected-error{{cannot take address of field referred to by __counted_by on a flexible array member}}
  return &v->buf; // expected-error{{cannot take address of incomplete __counted_by array}}
                  // expected-note@-1{{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(len)]'}}
  return &(v->buf+2); // expected-error{{cannot take the address of an rvalue of type 'int *__bidi_indexable'}}
}

int main() {
  struct S s = {0};
  // expected-error@+1{{initializing 't.buf' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct T t = {0};
  struct U u = {0};

  int local_len = 10;
  int *__single ptr_to_len = &s.len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **ptr_to_buf = &s.buf;         // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **ptr_ptr_to_len = &ptr_to_len;
  ptr_to_buf = &s.buf;  // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  *ptr_to_len = &s.len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  struct S *struct_ptr = &s;
  *ptr_to_len = &struct_ptr->len;   // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  *ptr_to_len = &(*struct_ptr).len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}

  *ptr_to_len = 100;

  foo(&s.len, &s.buf);
  foo(&local_len, &s.buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
  foo(&t.len, &t.buf);     // expected-error{{incompatible count expression '*out_len' vs. 'len + 1' in argument to function}}
  // expected-error@+1{{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  foo(&u.len, &u.buf);
  bar(&s.len, &s.buf); // expected-error{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}

  return 0;
}
