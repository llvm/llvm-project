
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  int *__counted_by(len) buf;
  int len;
};

struct S_Nullable {
  int *__counted_by_or_null(len) buf;
  int len;
};

struct T {
  int *__counted_by(len + 1) buf;
  int len;
};

struct T_Nullable {
  int *__counted_by_or_null(len + 1) buf;
  int len;
};

struct U {
  int *__counted_by(len) buf;
  int *__counted_by(len) buf2;
  int len;
};

struct U_Nullable {
  int *__counted_by_or_null(len) buf;
  int *__counted_by_or_null(len) buf2;
  int len;
};

struct V {
  int *__sized_by(len) buf;
  int *__counted_by(len) buf2;
  int *__indexable buf3;
  int *__bidi_indexable buf4;
  int *__single buf5;
  int *__sized_by_or_null(len) buf6;
  int *__counted_by_or_null(len) buf7;

  int len;
};

int arr[10];

void foo(int *out_len, int *__counted_by(*out_len) * out_buf) {
  // expected-note@-1 4{{passing argument to parameter 'out_buf' here}}
  *out_len = 9;
  *out_buf = arr;
  return;
}

void foo_nullable(int *out_len, int *__counted_by_or_null(*out_len) * out_buf) {
  // expected-note@-1 4{{passing argument to parameter 'out_buf' here}}
  *out_len = 9;
  *out_buf = arr;
  return;
}

void bar(int *fake_out_len, int **fake_out_buf) {
  *fake_out_buf = arr;
  *fake_out_len = 12;
  return;
}

void bar_nullable(int *fake_out_len, int **fake_out_buf) {
  *fake_out_buf = arr;
  *fake_out_len = 12;
  return;
}

void baz123(int *__counted_by(*out_len) buf, int *out_len, int *fake_out_len);
void baz132(int *__counted_by(*out_len) buf, int *fake_out_len, int *out_len);
void baz213(int *out_len, int *__counted_by(*out_len) buf, int *fake_out_len);
void baz312(int *fake_out_len, int *__counted_by(*out_len) buf, int *out_len);
void bazo123(int *__counted_by(*out_len) *out_buf, int *out_len, int *fake_out_len);
void bazo132(int *__counted_by(*out_len) *out_buf, int *fake_out_len, int *out_len);
void bazo213(int *out_len, int *__counted_by(*out_len) *out_buf, int *fake_out_len);
void bazo312(int *fake_out_len, int *__counted_by(*out_len) *out_buf, int *out_len);

void baz123_nullable(int *__counted_by_or_null(*out_len) buf, int *out_len, int *fake_out_len);
void baz132_nullable(int *__counted_by_or_null(*out_len) buf, int *fake_out_len, int *out_len);
void baz213_nullable(int *out_len, int *__counted_by_or_null(*out_len) buf, int *fake_out_len);
void baz312_nullable(int *fake_out_len, int *__counted_by_or_null(*out_len) buf, int *out_len);
void bazo123_nullable(int *__counted_by_or_null(*out_len) *out_buf, int *out_len, int *fake_out_len);
void bazo132_nullable(int *__counted_by_or_null(*out_len) *out_buf, int *fake_out_len, int *out_len);
void bazo213_nullable(int *out_len, int *__counted_by_or_null(*out_len) *out_buf, int *fake_out_len);
void bazo312_nullable(int *fake_out_len, int *__counted_by_or_null(*out_len) *out_buf, int *out_len);

int main() {
  struct S s = {0};
  struct S_Nullable s_n = {0};
  // expected-error@+1{{initializing 't.buf' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct T t = {0};
  struct T_Nullable t_n = {0};
  struct U u = {0};
  struct U_Nullable u_n = {0};
  struct V v = {0};

  int local_len = 10;
  int *__single ptr_to_len = &s.len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **ptr_to_buf = &s.buf;         // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **ptr_ptr_to_len = &ptr_to_len;
  ptr_to_buf = &s.buf;  // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  *ptr_to_len = &s.len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  *ptr_to_len = &s_n.len; // expected-error{{field referred to by '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  struct S *struct_ptr = &s;
  *ptr_to_len = &struct_ptr->len;   // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  *ptr_to_len = &(*struct_ptr).len; // expected-error{{field referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}

  *ptr_to_len = 100;

  foo(&s.len, &s.buf);
  foo(&local_len, &s.buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
  foo(&t.len, &t.buf);     // expected-error{{incompatible count expression '*out_len' vs. 'len + 1' in argument to function}}
  foo(&u.len, &u.buf);
  // expected-error@-1{{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  bar(&s.len, &s.buf); // expected-error{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  foo(&v.len, &v.buf);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  foo(&v.len, &v.buf2);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}
  foo(&v.len, &v.buf3);
  // expected-error@-1 {{passing 'int *__indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by(*out_len)*__single' (aka 'int *__single*__single')}}
  foo(&v.len, &v.buf4);
  // expected-error@-1 {{passing 'int *__bidi_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by(*out_len)*__single' (aka 'int *__single*__single')}}
  foo(&v.len, &v.buf5);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}
  bar(&v.len, &v.buf);
  // expected-error@-1 {{passing address of 'len' referred to by '__sized_by' to a parameter that is not referred to by the same attribute}}
  bar(&v.len, &v.buf2);
  // expected-error@-1 {{passing address of 'len' referred to by '__sized_by' to a parameter that is not referred to by the same attribute}}
  foo(&v.len, &v.buf3);
  // expected-error@-1 {{passing 'int *__indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by(*out_len)*__single' (aka 'int *__single*__single')}}
  foo(&v.len, &v.buf4);
  // expected-error@-1 {{passing 'int *__bidi_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by(*out_len)*__single' (aka 'int *__single*__single')}}
  foo(&v.len, &v.buf5);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}

  foo(&s_n.len, &s_n.buf);
  foo(&local_len, &s_n.buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
  foo(&t_n.len, &t_n.buf);     // expected-error{{incompatible count expression '*out_len' vs. 'len + 1' in argument to function}}
  foo(&u_n.len, &u_n.buf);
  // expected-error@-1{{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by_or_null(len)' (aka 'int *__single'), refers to 'len'}}
  bar(&s_n.len, &s_n.buf); // expected-error{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  foo_nullable(&s.len, &s.buf);
  foo_nullable(&local_len, &s.buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
  foo_nullable(&t.len, &t.buf);     // expected-error{{incompatible count expression '*out_len' vs. 'len + 1' in argument to function}}
  foo_nullable(&u.len, &u.buf);
  // expected-error@-1{{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  bar_nullable(&s.len, &s.buf); // expected-error{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  foo_nullable(&v.len, &v.buf);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  foo_nullable(&v.len, &v.buf2);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}
  foo_nullable(&v.len, &v.buf3);
  // expected-error@-1 {{passing 'int *__indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by_or_null(*out_len)*__single' (aka 'int *__single*__single')}}
  foo_nullable(&v.len, &v.buf4);
  // expected-error@-1 {{passing 'int *__bidi_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by_or_null(*out_len)*__single' (aka 'int *__single*__single')}}
  foo_nullable(&v.len, &v.buf5);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}
  bar_nullable(&v.len, &v.buf);
  // expected-error@-1 {{passing address of 'len' referred to by '__sized_by' to a parameter that is not referred to by the same attribute}}
  bar_nullable(&v.len, &v.buf2);
  // expected-error@-1 {{passing address of 'len' referred to by '__sized_by' to a parameter that is not referred to by the same attribute}}
  foo_nullable(&v.len, &v.buf3);
  // expected-error@-1 {{passing 'int *__indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by_or_null(*out_len)*__single' (aka 'int *__single*__single')}}
  foo_nullable(&v.len, &v.buf4);
  // expected-error@-1 {{passing 'int *__bidi_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single __counted_by_or_null(*out_len)*__single' (aka 'int *__single*__single')}}
  foo_nullable(&v.len, &v.buf5);
  // expected-error@-1 {{passing address of 'len' as an indirect parameter; must also pass 'buf' or its address because the type of 'buf', 'int *__single __sized_by(len)' (aka 'int *__single'), refers to 'len'}}

  foo_nullable(&s_n.len, &s_n.buf);
  foo_nullable(&local_len, &s_n.buf); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
  foo_nullable(&t_n.len, &t_n.buf);     // expected-error{{incompatible count expression '*out_len' vs. 'len + 1' in argument to function}}
  foo_nullable(&u_n.len, &u_n.buf);
  // expected-error@-1{{passing address of 'len' as an indirect parameter; must also pass 'buf2' or its address because the type of 'buf2', 'int *__single __counted_by_or_null(len)' (aka 'int *__single'), refers to 'len'}}
  bar_nullable(&s_n.len, &s_n.buf); // expected-error{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  baz123(s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz132(s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz213(&s.len, s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz312(&s.len, s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}

  baz123(s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz132(s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz213(&s_n.len, s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz312(&s_n.len, s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  bazo123(&s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo132(&s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo213(&s.len, &s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo312(&s.len, &s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}

  bazo123(&s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo132(&s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo213(&s_n.len, &s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo312(&s_n.len, &s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  baz123_nullable(s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz132_nullable(s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz213_nullable(&s.len, s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  baz312_nullable(&s.len, s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}

  bazo123_nullable(&s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo132_nullable(&s.buf, &s.len, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo213_nullable(&s.len, &s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  bazo312_nullable(&s.len, &s.buf, &s.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}

  baz123_nullable(s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz132_nullable(s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz213_nullable(&s_n.len, s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  baz312_nullable(&s_n.len, s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  bazo123_nullable(&s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo132_nullable(&s_n.buf, &s_n.len, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo213_nullable(&s_n.len, &s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}
  bazo312_nullable(&s_n.len, &s_n.buf, &s_n.len);
  // expected-error@-1{{passing address of 'len' referred to by '__counted_by_or_null' to a parameter that is not referred to by the same attribute}}

  return 0;
}
