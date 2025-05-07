
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// __unsafe_indexable

struct unsafe_indexable_struct {
  int *_Atomic __unsafe_indexable p1;
  int *__unsafe_indexable _Atomic p2;
  _Atomic(int *__unsafe_indexable) p3;
  _Atomic(int *) __unsafe_indexable p4;
};

void unsafe_indexable_local(void) {
  int *_Atomic __unsafe_indexable p1;
  int *__unsafe_indexable _Atomic p2;
  _Atomic(int *__unsafe_indexable) p3;
  _Atomic(int *) __unsafe_indexable p4;

  int *_Atomic __unsafe_indexable *_Atomic __unsafe_indexable p5;
  int *__unsafe_indexable _Atomic *_Atomic __unsafe_indexable p6;
  _Atomic(int *__unsafe_indexable) *_Atomic __unsafe_indexable p7;
  _Atomic(int *) __unsafe_indexable *_Atomic __unsafe_indexable p8;
}

int *_Atomic __unsafe_indexable unsafe_indexable_decl_ret1(void);
int *__unsafe_indexable _Atomic unsafe_indexable_decl_ret2(void);
_Atomic(int *__unsafe_indexable) unsafe_indexable_decl_ret3(void);
_Atomic(int *) __unsafe_indexable unsafe_indexable_decl_ret4(void);

// expected-warning@+1{{non-void function does not return a value}}
int *_Atomic __unsafe_indexable unsafe_indexable_def_ret1(void) {}
// expected-warning@+1{{non-void function does not return a value}}
int *__unsafe_indexable _Atomic unsafe_indexable_def_ret2(void) {}
// expected-warning@+1{{non-void function does not return a value}}
_Atomic(int *__unsafe_indexable) unsafe_indexable_def_ret3(void) {}
// expected-warning@+1{{non-void function does not return a value}}
_Atomic(int *) __unsafe_indexable unsafe_indexable_def_ret4(void) {}

void unsafe_indexable_decl_p1(int *_Atomic __unsafe_indexable p);
void unsafe_indexable_decl_p2(int *__unsafe_indexable _Atomic p);
void unsafe_indexable_decl_p3(_Atomic(int *__unsafe_indexable) p);
void unsafe_indexable_decl_p4(_Atomic(int *) __unsafe_indexable p);

void unsafe_indexable_def_p1(int *_Atomic __unsafe_indexable p) {}
void unsafe_indexable_def_p2(int *__unsafe_indexable _Atomic p) {}
void unsafe_indexable_def_p3(_Atomic(int *__unsafe_indexable) p) {}
void unsafe_indexable_def_p4(_Atomic(int *) __unsafe_indexable p) {}

// __single

struct single_struct {
  int *_Atomic __single p1;
  int *__single _Atomic p2;
  _Atomic(int *__single) p3;
  _Atomic(int *) __single p4;
};

void single_local(void) {
  int *_Atomic __single p1;
  int *__single _Atomic p2;
  _Atomic(int *__single) p3;
  _Atomic(int *) __single p4;

  int *_Atomic __single *_Atomic __single p5;
  int *__single _Atomic *_Atomic __single p6;
  _Atomic(int *__single) *_Atomic __single p7;
  _Atomic(int *) __single *_Atomic __single p8;
}
int *_Atomic __single single_decl_ret1(void);
int *__single _Atomic single_decl_ret2(void);
_Atomic(int *__single) single_decl_ret3(void);
_Atomic(int *) __single single_decl_ret4(void);

// expected-warning@+1{{non-void function does not return a value}}
int *_Atomic __single single_def_ret1(void) {}
// expected-warning@+1{{non-void function does not return a value}}
int *__single _Atomic single_def_ret2(void) {}
// expected-warning@+1{{non-void function does not return a value}}
_Atomic(int *__single) single_def_ret3(void) {}
// expected-warning@+1{{non-void function does not return a value}}
_Atomic(int *) __single single_def_ret4(void) {}

void single_decl_p1(int *_Atomic __single p);
void single_decl_p2(int *__single _Atomic p);
void single_decl_p3(_Atomic(int *__single) p);
void single_decl_p4(_Atomic(int *) __single p);

void single_def_p1(int *_Atomic __single p) {}
void single_def_p2(int *__single _Atomic p) {}
void single_def_p3(_Atomic(int *__single) p) {}
void single_def_p4(_Atomic(int *) __single p) {}

// __indexable

struct indexable_struct {
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *_Atomic __indexable p1;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *__indexable _Atomic p2;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *__indexable) p3;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *) __indexable p4;
};

void indexable_local(void) {
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *_Atomic __indexable p1;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *__indexable _Atomic p2;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *__indexable) p3;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *) __indexable p4;

  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *_Atomic __indexable *_Atomic __unsafe_indexable p5;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *__indexable _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *__indexable) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *) __indexable *_Atomic __unsafe_indexable p8;

  // expected-error@+2{{_Atomic on '__indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *_Atomic __indexable *_Atomic __indexable p9;
  // expected-error@+2{{_Atomic on '__indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  int *__indexable _Atomic *_Atomic __indexable p10;
  // expected-error@+2{{_Atomic on '__indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *__indexable) *_Atomic __indexable p11;
  // expected-error@+2{{_Atomic on '__indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *) __indexable *_Atomic __indexable p12;

  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int *_Nullable __indexable) p13;
  // expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
  _Atomic(int * _Nullable) __indexable p14;
}

// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
int *_Atomic __indexable indexable_decl_ret1(void);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
int *__indexable _Atomic indexable_decl_ret2(void);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
_Atomic(int *__indexable) indexable_decl_ret3(void);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
_Atomic(int *) __indexable indexable_decl_ret4(void);

// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
int *_Atomic __indexable indexable_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
int *__indexable _Atomic indexable_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
_Atomic(int *__indexable) indexable_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
_Atomic(int *) __indexable indexable_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_decl_p1(int *_Atomic __indexable p);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_decl_p2(int *__indexable _Atomic p);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_decl_p3(_Atomic(int *__indexable) p);
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_decl_p4(_Atomic(int *) __indexable p);

// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_def_p1(int *_Atomic __indexable p) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_def_p2(int *__indexable _Atomic p) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_def_p3(_Atomic(int *__indexable) p) {}
// expected-error@+1{{_Atomic on '__indexable' pointer is not yet supported}}
void indexable_def_p4(_Atomic(int *) __indexable p) {}

// __bidi_indexable

struct bidi_indexable_struct {
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *_Atomic __bidi_indexable p1;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *__bidi_indexable _Atomic p2;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *__bidi_indexable) p3;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *) __bidi_indexable p4;
};

void bidi_indexable_local(void) {
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *_Atomic __bidi_indexable p1;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *__bidi_indexable _Atomic p2;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *__bidi_indexable) p3;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *) __bidi_indexable p4;

  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *_Atomic __bidi_indexable *_Atomic __unsafe_indexable p5;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *__bidi_indexable _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *__bidi_indexable) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *) __bidi_indexable *_Atomic __unsafe_indexable p8;

  // expected-error@+2{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *_Atomic __bidi_indexable *_Atomic __bidi_indexable p9;
  // expected-error@+2{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *__bidi_indexable _Atomic *_Atomic __bidi_indexable p10;
  // expected-error@+2{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *__bidi_indexable) *_Atomic __bidi_indexable p11;
  // expected-error@+2{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *) __bidi_indexable *_Atomic __bidi_indexable p12;
}

// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
int *_Atomic __bidi_indexable bidi_indexable_decl_ret1(void);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
int *__bidi_indexable _Atomic bidi_indexable_decl_ret2(void);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
_Atomic(int *__bidi_indexable) bidi_indexable_decl_ret3(void);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
_Atomic(int *) __bidi_indexable bidi_indexable_decl_ret4(void);

// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
int *_Atomic __bidi_indexable bidi_indexable_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
int *__bidi_indexable _Atomic bidi_indexable_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
_Atomic(int *__bidi_indexable) bidi_indexable_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
_Atomic(int *) __bidi_indexable bidi_indexable_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_decl_p1(int *_Atomic __bidi_indexable p);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_decl_p2(int *__bidi_indexable _Atomic p);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_decl_p3(_Atomic(int *__bidi_indexable) p);
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_decl_p4(_Atomic(int *) __bidi_indexable p);

// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_def_p1(int *_Atomic __bidi_indexable p) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_def_p2(int *__bidi_indexable _Atomic p) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_def_p3(_Atomic(int *__bidi_indexable) p) {}
// expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
void bidi_indexable_def_p4(_Atomic(int *) __bidi_indexable p) {}

// __counted_by

struct counted_by_struct {
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *_Atomic __counted_by(16) p1;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *__counted_by(16) _Atomic p2;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *__counted_by(16)) p3;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *) __counted_by(16) p4;
};

void counted_by_local(void) {
  int len;

  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *_Atomic __counted_by(len) p1;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *__counted_by(len) _Atomic p2;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *__counted_by(len)) p3;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *) __counted_by(len) p4;

  // expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *_Atomic __counted_by(len) * _Atomic __unsafe_indexable p5;
  // expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__counted_by(len) _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *__counted_by(len)) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *) __counted_by(len) * _Atomic __unsafe_indexable p8;
}

// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
int *_Atomic __counted_by(16) counted_by_decl_ret1(void);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
int *__counted_by(16) _Atomic counted_by_decl_ret2(void);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}c
_Atomic(int *__counted_by(16)) counted_by_decl_ret3(void);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
_Atomic(int *) __counted_by(16) counted_by_decl_ret4(void);

// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
int *_Atomic __counted_by(16) counted_by_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
int *__counted_by(16) _Atomic counted_by_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
_Atomic(int *__counted_by(16)) counted_by_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
_Atomic(int *) __counted_by(16) counted_by_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_decl_p1(int *_Atomic __counted_by(16) p);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_decl_p2(int *__counted_by(16) _Atomic p);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_decl_p3(_Atomic(int *__counted_by(16)) p);
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_decl_p4(_Atomic(int *) __counted_by(16) p);

// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_def_p1(int *_Atomic __counted_by(16) p) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_def_p2(int *__counted_by(16) _Atomic p) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_def_p3(_Atomic(int *__counted_by(16)) p) {}
// expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
void counted_by_def_p4(_Atomic(int *) __counted_by(16) p) {}

// __sized_by

struct sized_by_struct {
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  int *_Atomic __sized_by(16) p1;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  int *__sized_by(16) _Atomic p2;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *__sized_by(16)) p3;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *) __sized_by(16) p4;
};

void sized_by_local(void) {
  int size;

  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  int *_Atomic __sized_by(size) p1;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  int *__sized_by(size) _Atomic p2;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *__sized_by(size)) p3;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *) __sized_by(size) p4;

  // expected-error@+1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *_Atomic __sized_by(size) * _Atomic __unsafe_indexable p5;
  // expected-error@+1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__sized_by(size) _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *__sized_by(size)) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
  _Atomic(int *) __sized_by(size) * _Atomic __unsafe_indexable p8;
}

// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
int *_Atomic __sized_by(16) sized_by_decl_ret1(void);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
int *__sized_by(16) _Atomic sized_by_decl_ret2(void);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
_Atomic(int *__sized_by(16)) sized_by_decl_ret3(void);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
_Atomic(int *) __sized_by(16) sized_by_decl_ret4(void);

// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
int *_Atomic __sized_by(16) sized_by_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
int *__sized_by(16) _Atomic sized_by_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
_Atomic(int *__sized_by(16)) sized_by_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
_Atomic(int *) __sized_by(16) sized_by_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_decl_p1(int *_Atomic __sized_by(16) p);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_decl_p2(int *__sized_by(16) _Atomic p);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_decl_p3(_Atomic(int *__sized_by(16)) p);
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_decl_p4(_Atomic(int *) __sized_by(16) p);

// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_def_p1(int *_Atomic __sized_by(16) p) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_def_p2(int *__sized_by(16) _Atomic p) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_def_p3(_Atomic(int *__sized_by(16)) p) {}
// expected-error@+1{{_Atomic on '__sized_by' pointer is not yet supported}}
void sized_by_def_p4(_Atomic(int *) __sized_by(16) p) {}

// __counted_by_or_null

struct counted_by_or_null_struct {
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  int *_Atomic __counted_by_or_null(16) p1;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  int *__counted_by_or_null(16) _Atomic p2;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *__counted_by_or_null(16)) p3;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __counted_by_or_null(16) p4;
};

void counted_by_or_null_local(void) {
  int len;

  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  int *_Atomic __counted_by_or_null(len) p1;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  int *__counted_by_or_null(len) _Atomic p2;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *__counted_by_or_null(len)) p3;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __counted_by_or_null(len) p4;

  // expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
  int *_Atomic __counted_by_or_null(len) * _Atomic __unsafe_indexable p5;
  // expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__counted_by_or_null(len) _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *__counted_by_or_null(len)) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __counted_by_or_null(len) * _Atomic __unsafe_indexable p8;
}

// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
int *_Atomic __counted_by_or_null(16) counted_by_or_null_decl_ret1(void);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
int *__counted_by_or_null(16) _Atomic counted_by_or_null_decl_ret2(void);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}c
_Atomic(int *__counted_by_or_null(16)) counted_by_or_null_decl_ret3(void);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
_Atomic(int *) __counted_by_or_null(16) counted_by_or_null_decl_ret4(void);

// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
int *_Atomic __counted_by_or_null(16) counted_by_or_null_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
int *__counted_by_or_null(16) _Atomic counted_by_or_null_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
_Atomic(int *__counted_by_or_null(16)) counted_by_or_null_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
_Atomic(int *) __counted_by_or_null(16) counted_by_or_null_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_decl_p1(int *_Atomic __counted_by_or_null(16) p);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_decl_p2(int *__counted_by_or_null(16) _Atomic p);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_decl_p3(_Atomic(int *__counted_by_or_null(16)) p);
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_decl_p4(_Atomic(int *) __counted_by_or_null(16) p);

// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_def_p1(int *_Atomic __counted_by_or_null(16) p) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_def_p2(int *__counted_by_or_null(16) _Atomic p) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_def_p3(_Atomic(int *__counted_by_or_null(16)) p) {}
// expected-error@+1{{_Atomic on '__counted_by_or_null' pointer is not yet supported}}
void counted_by_or_null_def_p4(_Atomic(int *) __counted_by_or_null(16) p) {}

// __sized_by_or_null

struct sized_by_or_null_struct {
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  int *_Atomic __sized_by_or_null(16) p1;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  int *__sized_by_or_null(16) _Atomic p2;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *__sized_by_or_null(16)) p3;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __sized_by_or_null(16) p4;
};

void sized_by_or_null_local(void) {
  int size;

  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  int *_Atomic __sized_by_or_null(size) p1;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  int *__sized_by_or_null(size) _Atomic p2;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *__sized_by_or_null(size)) p3;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __sized_by_or_null(size) p4;

  // expected-error@+1{{'__sized_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
  int *_Atomic __sized_by_or_null(size) * _Atomic __unsafe_indexable p5;
  // expected-error@+1{{'__sized_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__sized_by_or_null(size) _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *__sized_by_or_null(size)) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
  _Atomic(int *) __sized_by_or_null(size) * _Atomic __unsafe_indexable p8;
}

// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
int *_Atomic __sized_by_or_null(16) sized_by_or_null_decl_ret1(void);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
int *__sized_by_or_null(16) _Atomic sized_by_or_null_decl_ret2(void);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
_Atomic(int *__sized_by_or_null(16)) sized_by_or_null_decl_ret3(void);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
_Atomic(int *) __sized_by_or_null(16) sized_by_or_null_decl_ret4(void);

// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
int *_Atomic __sized_by_or_null(16) sized_by_or_null_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
int *__sized_by_or_null(16) _Atomic sized_by_or_null_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
_Atomic(int *__sized_by_or_null(16)) sized_by_or_null_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
_Atomic(int *) __sized_by_or_null(16) sized_by_or_null_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_decl_p1(int *_Atomic __sized_by_or_null(16) p);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_decl_p2(int *__sized_by_or_null(16) _Atomic p);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_decl_p3(_Atomic(int *__sized_by_or_null(16)) p);
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_decl_p4(_Atomic(int *) __sized_by_or_null(16) p);

// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_def_p1(int *_Atomic __sized_by_or_null(16) p) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_def_p2(int *__sized_by_or_null(16) _Atomic p) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_def_p3(_Atomic(int *__sized_by_or_null(16)) p) {}
// expected-error@+1{{_Atomic on '__sized_by_or_null' pointer is not yet supported}}
void sized_by_or_null_def_p4(_Atomic(int *) __sized_by_or_null(16) p) {}

// __ended_by

void ended_by_params(
  int * pEnd,

  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  int *_Atomic __ended_by(pEnd) p1,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  int *__ended_by(pEnd) _Atomic p2,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  _Atomic(int *__ended_by(pEnd)) p3,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  _Atomic(int *) __ended_by(pEnd) p4,

  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  int *_Atomic __ended_by(pEnd) * _Atomic __unsafe_indexable p5,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  int *__ended_by(pEnd) _Atomic *_Atomic __unsafe_indexable p6,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  _Atomic(int *__ended_by(pEnd)) *_Atomic __unsafe_indexable p7,
  // expected-error@+1{{_Atomic on '__ended_by' pointer is not yet supported}}
  _Atomic(int *) __ended_by(pEnd) * _Atomic __unsafe_indexable p8,

  int *_Atomic  p9,
  // expected-error@+1{{_Atomic on 'end' pointer is not yet supported}}
  int * __ended_by(p9) pStart,
  int *_Atomic  * _Atomic p10,
  // expected-error@+1{{_Atomic on 'end' pointer is not yet supported}}
  int * __ended_by(p10) pStart2,
  int * * _Atomic p11,
  // expected-error@+1{{_Atomic on 'end' pointer is not yet supported}}
  int * __ended_by(p11) pStart3,
  int * _Atomic * p12,
  int * * __ended_by(p12) pStart4
);

// __terminated_by

struct terminated_by_struct {
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *_Atomic __null_terminated __single p1;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *__null_terminated __single _Atomic p2;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *__null_terminated __single) p3;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *) __null_terminated __single p4;
};

void terminated_by_local(void) {
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *_Atomic __null_terminated __single p1;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *__null_terminated __single _Atomic p2;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *__null_terminated __single) p3;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *) __null_terminated __single p4;

  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *_Atomic __null_terminated __single *_Atomic __unsafe_indexable p5;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *__null_terminated __single _Atomic *_Atomic __unsafe_indexable p6;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *__null_terminated __single) *_Atomic __unsafe_indexable p7;
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *) __null_terminated __single *_Atomic __unsafe_indexable p8;

  // expected-error@+2{{_Atomic on '__terminated_by' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *_Atomic __null_terminated __single *_Atomic __null_terminated __single p9;
  // expected-error@+2{{_Atomic on '__terminated_by' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  int *__null_terminated __single _Atomic *_Atomic __null_terminated __single p10;
  // expected-error@+2{{_Atomic on '__terminated_by' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *__null_terminated __single) *_Atomic __null_terminated __single p11;
  // expected-error@+2{{_Atomic on '__terminated_by' pointer is not yet supported}}
  // expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
  _Atomic(int *) __null_terminated __single *_Atomic __null_terminated __single p12;
}

// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
int *_Atomic __null_terminated __single terminated_by_decl_ret1(void);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
int *__null_terminated __single _Atomic terminated_by_decl_ret2(void);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
_Atomic(int *__null_terminated __single) terminated_by_decl_ret3(void);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
_Atomic(int *) __null_terminated __single terminated_by_decl_ret4(void);

// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
int *_Atomic __null_terminated __single terminated_by_def_ret1(void) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
int *__null_terminated __single _Atomic terminated_by_def_ret2(void) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
_Atomic(int *__null_terminated __single) terminated_by_def_ret3(void) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
_Atomic(int *) __null_terminated __single terminated_by_def_ret4(void) {}

// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_decl_p1(int *_Atomic __null_terminated __single p);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_decl_p2(int *__null_terminated __single _Atomic p);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_decl_p3(_Atomic(int *__null_terminated __single) p);
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_decl_p4(_Atomic(int *) __null_terminated __single p);

// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_def_p1(int *_Atomic __null_terminated __single p) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_def_p2(int *__null_terminated __single _Atomic p) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_def_p3(_Atomic(int *__null_terminated __single) p) {}
// expected-error@+1{{_Atomic on '__terminated_by' pointer is not yet supported}}
void terminated_by_def_p4(_Atomic(int *) __null_terminated __single p) {}
